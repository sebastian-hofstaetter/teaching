#
# core evaluation metrics computation
# ------------------------------------------------------------
#

import numpy as np
import re
import string
from collections import Counter

global_metric_config = {
    "MRR+Recall@": [10, 20, 1000],  # multiple allowed
    "nDCG@": [3, 5, 10, 20, 1000],  # multiple allowed
    "MAP@": 1000,  # only one allowed
}


#
# calculate ranked retrieval result metrics
#

def calculate_metrics_plain(ranking, qrels, binarization_point=1.0, return_per_query=False):
    """
    calculate main evaluation metrics for the given results (without looking at candidates),
    returns a dict of metrics
    """

    rank_padding_max = max(max(x) if type(x) == list else x for x in global_metric_config.values()) + 1

    ranked_queries = len(ranking)
    ap_per_candidate_depth = np.zeros(ranked_queries)
    rr_per_candidate_depth = np.zeros((len(global_metric_config["MRR+Recall@"]), ranked_queries))
    rank_per_candidate_depth = np.zeros((len(global_metric_config["MRR+Recall@"]), ranked_queries))
    recall_per_candidate_depth = np.zeros((len(global_metric_config["MRR+Recall@"]), ranked_queries))
    ndcg_per_candidate_depth = np.zeros((len(global_metric_config["nDCG@"]), ranked_queries))
    evaluated_queries = 0

    for query_index, (query_id, ranked_doc_ids) in enumerate(ranking.items()):
        if query_id in qrels:
            evaluated_queries += 1

            relevant_ids = np.array(list(qrels[query_id].keys()))  # key, value guaranteed in same order
            relevant_grades = np.array(list(qrels[query_id].values()))
            sorted_relevant_grades = np.sort(relevant_grades)[::-1]

            num_relevant = relevant_ids.shape[0]
            np_rank = np.array(ranked_doc_ids)
            relevant_mask = np.in1d(np_rank, relevant_ids)  # shape: (ranking_depth,) - type: bool

            binary_relevant = relevant_ids[relevant_grades >= binarization_point]
            binary_num_relevant = binary_relevant.shape[0]
            binary_relevant_mask = np.in1d(np_rank, binary_relevant)  # shape: (ranking_depth,) - type: bool

            # check if we have a relevant document at all in the results -> if not skip and leave 0 
            if np.any(binary_relevant_mask):

                # now select the relevant ranks across the fixed ranks
                ranks = np.arange(1, binary_relevant_mask.shape[0] + 1)[binary_relevant_mask]

                #
                # ap
                #
                map_ranks = ranks[ranks <= global_metric_config["MAP@"]]
                ap = np.arange(1, map_ranks.shape[0] + 1) / map_ranks
                ap = np.sum(ap) / binary_num_relevant
                ap_per_candidate_depth[query_index] = ap

                # mrr only the first relevant rank is used
                first_rank = ranks[0]

                for cut_indx, cutoff in enumerate(global_metric_config["MRR+Recall@"]):

                    curr_ranks = ranks.copy()
                    curr_ranks[curr_ranks > cutoff] = 0

                    recall = (curr_ranks > 0).sum(axis=0) / binary_num_relevant
                    recall_per_candidate_depth[cut_indx, query_index] = recall

                    #
                    # mrr
                    #

                    # ignore ranks that are out of the interest area (leave 0)
                    if first_rank <= cutoff:
                        rr_per_candidate_depth[cut_indx, query_index] = 1 / first_rank
                        rank_per_candidate_depth[cut_indx, query_index] = first_rank

            if np.any(relevant_mask):

                # now select the relevant ranks across the fixed ranks
                ranks = np.arange(1, relevant_mask.shape[0] + 1)[relevant_mask]

                grades_per_rank = np.ndarray(ranks.shape[0], dtype=int)
                for i, id in enumerate(np_rank[relevant_mask]):
                    grades_per_rank[i] = np.where(relevant_ids == id)[0]

                grades_per_rank = relevant_grades[grades_per_rank]

                #
                # ndcg = dcg / idcg 
                #
                for cut_indx, cutoff in enumerate(global_metric_config["nDCG@"]):
                    #
                    # get idcg (from relevant_ids)
                    idcg = (sorted_relevant_grades[:cutoff] / np.log2(1 + np.arange(1, min(num_relevant, cutoff) + 1)))

                    curr_ranks = ranks.copy()
                    curr_ranks[curr_ranks > cutoff] = 0

                    with np.errstate(divide='ignore', invalid='ignore'):
                        c = np.true_divide(grades_per_rank, np.log2(1 + curr_ranks))
                        c[c == np.inf] = 0
                        dcg = np.nan_to_num(c)

                    nDCG = dcg.sum(axis=-1) / idcg.sum()

                    ndcg_per_candidate_depth[cut_indx, query_index] = nDCG

    mrr = rr_per_candidate_depth.sum(axis=-1) / evaluated_queries
    relevant = (rr_per_candidate_depth > 0).sum(axis=-1)
    non_relevant = (rr_per_candidate_depth == 0).sum(axis=-1)

    avg_rank = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), -1, rank_per_candidate_depth)
    avg_rank[np.isnan(avg_rank)] = 0.

    median_rank = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), -1, rank_per_candidate_depth)
    median_rank[np.isnan(median_rank)] = 0.

    map_score = ap_per_candidate_depth.sum(axis=-1) / evaluated_queries
    recall = recall_per_candidate_depth.sum(axis=-1) / evaluated_queries
    nDCG = ndcg_per_candidate_depth.sum(axis=-1) / evaluated_queries

    local_dict = {}

    for cut_indx, cutoff in enumerate(global_metric_config["MRR+Recall@"]):
        local_dict['MRR@' + str(cutoff)] = mrr[cut_indx]
        local_dict['Recall@' + str(cutoff)] = recall[cut_indx]
        local_dict['QueriesWithNoRelevant@' + str(cutoff)] = non_relevant[cut_indx]
        local_dict['QueriesWithRelevant@' + str(cutoff)] = relevant[cut_indx]
        local_dict['AverageRankGoldLabel@' + str(cutoff)] = avg_rank[cut_indx]
        local_dict['MedianRankGoldLabel@' + str(cutoff)] = median_rank[cut_indx]

    for cut_indx, cutoff in enumerate(global_metric_config["nDCG@"]):
        local_dict['nDCG@' + str(cutoff)] = nDCG[cut_indx]

    local_dict['QueriesRanked'] = evaluated_queries
    local_dict['MAP@' + str(global_metric_config["MAP@"])] = map_score

    if return_per_query:
        return local_dict, rr_per_candidate_depth, ap_per_candidate_depth, recall_per_candidate_depth, ndcg_per_candidate_depth
    else:
        return local_dict


# unrolled: dict<qid,[(did,score)]>
# returns: dict<qid,[did]> where list of docIds is sorted by score descending
def unrolled_to_ranked_result(unrolled_results):
    ranked_result = {}
    for query_id, query_data in unrolled_results.items():
        local_list = []
        # sort the results per query based on the output
        for (doc_id, output_value) in sorted(query_data, key=lambda x: x[1], reverse=True):
            local_list.append(doc_id)
        ranked_result[query_id] = local_list
    return ranked_result


#
# QA metrics
# source official squad eval:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
#


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


#
# use to get exact (normalized) overlap for 1 answer and 1 gold label
#
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


#
# use to get f1 (normalized) overlap for 1 answer and 1 gold label
#
def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


#
# I/O
#

def load_qrels(path):
    with open(path, 'r') as file:
        qids_to_relevant_passageids = {}
        for line in file:
            try:
                line = line.strip().split()
                qid = line[0]
                if float(line[3]) > 0.0001:
                    if qid not in qids_to_relevant_passageids:
                        qids_to_relevant_passageids[qid] = {}
                    qids_to_relevant_passageids[qid][line[2]] = float(line[3])
            except Exception:
                raise IOError('\"%s\" is not valid format' % line)
        return qids_to_relevant_passageids


def load_ranking(path, qrels=None):
    with open(path, 'r') as file:
        qid_to_ranked_candidate_passages = {}
        for line in file:
            try:
                line = line.strip().split()
                if qrels is not None:
                    if line[0] not in qrels:
                        continue

                if len(line) == 4:  # own format
                    qid = line[0]
                    pid = line[1].strip()
                if len(line) == 6:  # original trec format
                    qid = line[0]
                    pid = line[2].strip()
                if qid not in qid_to_ranked_candidate_passages:
                    qid_to_ranked_candidate_passages[qid] = []
                qid_to_ranked_candidate_passages[qid].append(pid)
            except Exception:
                raise IOError('\"%s\" is not valid format' % line)
        return qid_to_ranked_candidate_passages


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 3:
        metrics = calculate_metrics_plain(load_ranking(sys.argv[2]), load_qrels(sys.argv[1]), binarization_point=1)
        print('#####################')
        for metric in metrics:
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')
    else:
        print('Usage: <qrel> <output>')
