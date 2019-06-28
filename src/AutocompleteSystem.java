import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class AutocompleteSystem {

  private TrieWithHotness trie;
  private String curQuery;

  public AutocompleteSystem(String[] sentences, int[] times) {
    trie = new TrieWithHotness();
    for (int i = 0; i < sentences.length; i++) {
      trie.insertWithHotness(sentences[i], times[i]);
    }
    curQuery = "";
  }

  public List<String> input(char c) {
    if (c == '#') {
      if (!curQuery.equals("")) {
        trie.insertWithHotness(curQuery,1);
      }
      curQuery = "";
      return new ArrayList<>();
    } else {
      curQuery += c;
      List<TrieWithHotness.SentenceWithHotness> sentencesWithPrefix = trie.findAllSentencesWithPrefix(curQuery);
      sentencesWithPrefix.sort(new Comparator<TrieWithHotness.SentenceWithHotness>() {
        @Override
        public int compare(TrieWithHotness.SentenceWithHotness o1, TrieWithHotness.SentenceWithHotness o2) {
          if (o2.hotness == o1.hotness) {
            int comp = o1.sentence.compareTo(o2.sentence);
            if (comp == 0) {
              return o1.sentence.length() - o2.sentence.length();
            } else {
              return comp;
            }
          } else {
            return o2.hotness - o1.hotness;
          }
        }
      });
      List<String> ans = new ArrayList<>();
      for (int i = 0; i < sentencesWithPrefix.size() && i < 3; i++) {
        ans.add(sentencesWithPrefix.get(i).sentence);
      }
      return ans;
    }
  }

}



