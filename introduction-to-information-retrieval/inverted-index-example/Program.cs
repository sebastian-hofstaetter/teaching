using System;
using System.Diagnostics;
using System.IO;

/// <summary>
/// Runs a very simple indexer, we use only one sample doc over and over again (with simulated different document ids)
///
/// This is not a best practice example, as you will notice that the indexing gets slow (and uses a lot of memory)
/// when increasing the simulated document count.
///
/// InvertedIndexFaster shows some (but definitely not all possible) improvements over InvertedIndex
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        var stopwatch = Stopwatch.StartNew();

        var index = new InvertedIndexFaster();
        var totalTokens = 0;

        for (var i = 0; i < 3_000; i++)
        {
            var documentId = "doc-" + i;
            var content = File.ReadAllText("lincoln-wiki-entry.txt");
            var tokens = content.Split(' ');

            foreach (var token in tokens)
            {
                index.AddTerm(token, documentId);
                totalTokens++;
            }
        }

        Console.WriteLine($"Elapsed: {stopwatch.Elapsed} for {totalTokens} tokens");
    }
}