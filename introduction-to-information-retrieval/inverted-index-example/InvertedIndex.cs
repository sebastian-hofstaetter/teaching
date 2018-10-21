using System.Collections.Generic;
using System.Linq;


/// <summary>
/// Sample inverted index: this is not best practice, it is to show common pitfalls and things one can improve easily 
/// </summary>
public class InvertedIndex
{
    public Dictionary<string, List<PostingListEntry>> PostingLists { get; set; } // PostingListEntry is a reference type 
                                                                                 // the List (with backing array) just contains pointers to the different PostingListEntry
                                                                                 // objects: [pointer, pointer,...], they point to different memory locations
                                                                                 // BAD: iteration is not fast, structure increases pressure on the garbage collector, because it has to handle more objects 
    public InvertedIndex()
    {
        PostingLists = new Dictionary<string, List<PostingListEntry>>();
    }

    public void AddTerm(string term, string documentId)
    {
        if (PostingLists.ContainsKey(term))
        {
            var lastElement = PostingLists[term].Last(); // this is BAD - .Last() works on the IEnumerable interface and therefore 
                                                         // iterates through all documents all the time, but we know the length
                                                         // and can just jump there - see InvertedIndexFaster.cs
            if (lastElement.document == documentId)
            {
                PostingLists[term][PostingLists[term].Count - 1].frequency++; // PostingLists[term] is not free! it re-computes the hash and the position of the value
                                                                              // aka List<PostingListEntry> again, the compiler/framework does not know that we did not change anything
                                                                              // but we know that it must be the same list, so reuse it in a local variable,  - see InvertedIndexFaster.cs
            }
            else
            {
                PostingLists[term].Add(new PostingListEntry{document = documentId, frequency = 1}); // PostingLists[term] is not free! see above
            }
        }
        else
        {
            PostingLists[term] = new List<PostingListEntry> { new PostingListEntry { document = documentId, frequency = 1 } };
        }
    }
}
public class PostingListEntry
{
    public string document;
    public int frequency;
}