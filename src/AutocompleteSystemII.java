import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AutocompleteSystemII {

  private TrieWithHotness trie;
  private String curQuery;

  public AutocompleteSystemII(String[] sentences, int[] times) {
    trie = new TrieWithHotness();
    for (int i = 0; i < sentences.length; i++) {
      trie.insertWithHotness(sentences[i], times[i]);
    }
    curQuery = "";
  }

  public List<String> input(char c) {
    if (c == '#') {
      if (!curQuery.equals("")) {
        trie.insertWithHotness(curQuery, 1);
      }
      curQuery = "";
      return new ArrayList<>();
    } else {
      curQuery += c;
      return trie.findAllSentencesWithPrefix(curQuery);
    }
  }

  public class TrieWithHotness {

    private TrieNodeWithHotness root;

    public TrieWithHotness() {
      root = new TrieNodeWithHotness();
    }


    public void insertWithHotness(String word, int hotness) {
      TrieNodeWithHotness t = root;
      for (int i = 0; i < word.length(); i++) {
        int idx = getIdx(word.charAt(i));
        if (t.arr[idx] == null) {
          t.arr[idx] = new TrieNodeWithHotness();
        }
        t = t.arr[idx];
        t.counts.put(word, t.counts.getOrDefault(word, 0) + hotness);
      }
    }

    private int getIdx(char c) {
      return c == ' ' ? 26 : c - 'a';
    }


    public List<String> findAllSentencesWithPrefix(String word) {
      TrieNodeWithHotness t = root;
      for (int i = 0; i < word.length(); i++) {
        int idx = getIdx(word.charAt(i));
        if (t.arr[idx] == null) {
          return new ArrayList<>();
        }
        t = t.arr[idx];
      }
      List<SentenceWithHotness> sentencesWithPrefix = new ArrayList<>();
      for (String s : t.counts.keySet()) {
        sentencesWithPrefix.add(new SentenceWithHotness(s, t.counts.get(s)));
      }
      sentencesWithPrefix.sort((o1, o2) -> {
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
      });
      List<String> ans = new ArrayList<>();
      for (int i = 0; i < sentencesWithPrefix.size() && i < 3; i++) {
        ans.add(sentencesWithPrefix.get(i).sentence);
      }
      return ans;
    }

    class TrieNodeWithHotness {
      TrieNodeWithHotness[] arr;
      Map<String, Integer> counts;

      public TrieNodeWithHotness() {
        this.arr = new TrieNodeWithHotness[27];
        counts = new HashMap<>();
      }
    }

    class SentenceWithHotness {
      String sentence;
      int hotness;

      public SentenceWithHotness(String sentence, int hotness) {
        this.sentence = sentence;
        this.hotness = hotness;
      }
    }
  }

}



