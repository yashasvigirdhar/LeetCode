import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class TrieWithHotness {

  private TrieNodeWithHotness root;

  public TrieWithHotness() {
    root = new TrieNodeWithHotness();
  }


  public void insert(String word) {
    TrieNodeWithHotness t = root;
    for (int i = 0; i < word.length(); i++) {
      int idx = getIdx(word.charAt(i));
      if (t.arr[idx] == null) {
        t.arr[idx] = new TrieNodeWithHotness();
      }
      t = t.arr[idx];
    }
    t.hotness = t.hotness + 1;
    t.isEndOfWord = true;
  }

  public void insertWithHotness(String word, int hotness) {
    TrieNodeWithHotness t = root;
    for (int i = 0; i < word.length(); i++) {
      int idx = getIdx(word.charAt(i));
      if (t.arr[idx] == null) {
        t.arr[idx] = new TrieNodeWithHotness();
      }
      t = t.arr[idx];
    }
    t.hotness = hotness;
    t.isEndOfWord = true;
  }

  private int getIdx(char c) {
    return c == ' ' ? 26 : c - 'a';
  }


  public List<SentenceWithHotness> findAllSentencesWithPrefix(String word) {
    TrieNodeWithHotness t = root;
    for (int i = 0; i < word.length(); i++) {
      int idx = getIdx(word.charAt(i));
      if (t.arr[idx] == null) {
        return new ArrayList<>();
      }
      t = t.arr[idx];
    }
    List<SentenceWithHotness> allStringsFromThisNode = getAllStringsFromThisNode(t);
    List<SentenceWithHotness> ans = new ArrayList<>();
    for (SentenceWithHotness sentence : allStringsFromThisNode) {
      sentence.sentence = new StringBuilder().append(word).append(sentence.sentence).toString();
      ans.add(sentence);
    }
    return ans;
  }

  private List<SentenceWithHotness> getAllStringsFromThisNode(TrieNodeWithHotness t) {
    List<SentenceWithHotness> ans = new ArrayList<>();
    for (int i = 0; i < 27; i++) {
      if (t.arr[i] != null) {
        List<SentenceWithHotness> allStringsFromThisNode = getAllStringsFromThisNode(t.arr[i]);
        String stringToAdd = getStringToAdd(i);
        for (SentenceWithHotness sentence : allStringsFromThisNode) {
          sentence.sentence = stringToAdd + sentence.sentence;
          ans.add(sentence);
        }
      }
    }
    if (t.isEndOfWord) {
      ans.add(new SentenceWithHotness("", t.hotness));
    }
    return ans;
  }

  private String getStringToAdd(int i) {
    StringBuilder builder = new StringBuilder(1);
    if (i == 26) {
      builder.append(" ");
    } else {
      builder.append((char) ('a' + i));
    }
    return builder.toString();
  }

  public class TrieNodeWithHotness {
    TrieNodeWithHotness[] arr;
    int hotness;
    boolean isEndOfWord = false;

    public TrieNodeWithHotness() {
      this.arr = new TrieNodeWithHotness[27];
      this.hotness = 0;
    }
  }

  static public class SentenceWithHotness {
    String sentence;
    int hotness;

    public SentenceWithHotness(String sentence, int hotness) {
      this.sentence = sentence;
      this.hotness = hotness;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      SentenceWithHotness that = (SentenceWithHotness) o;
      return hotness == that.hotness &&
              Objects.equals(sentence, that.sentence);
    }

    @Override
    public int hashCode() {
      return Objects.hash(sentence, hotness);
    }
  }
}
