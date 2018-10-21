using System.Collections.Generic;

/// <summary>
/// Imporved sample inverted index: this is still not best practice, but at least some things are improved 
/// </summary>
public class InvertedIndexFaster
{
    // (string document, int frequency) is a value tuple which means the values (except for the string document, which should be an int for docID)
    // are saved inside the array structure and can be iterated over faster and use less memory + less pressure on the GC
    public Dictionary<string, List<(string document, int frequency)>> PostingLists { get; set; }

    public InvertedIndexFaster()
    {
        PostingLists = new Dictionary<string, List<(string, int)>>();
    }
    
    public void AddTerm(string term, string document)
    {
        var contains = PostingLists.TryGetValue(term, out var postingList); // combine contains check and get object reference 
        if (contains)
        {
            var lastElement = postingList[postingList.Count - 1]; // jump to last element with known length instead of linq method
            if (lastElement.document == document)
            {
                postingList[postingList.Count - 1] = (lastElement.document, lastElement.frequency + 1); // this does not allocate anything, we just have to write the tuple in one piece to the array
            }
            else
            {
                postingList.Add((document, 1));
            }
        }
        else
        {
            PostingLists[term] = new List<(string, int)> { (document, 1) };
        }
    }
}
