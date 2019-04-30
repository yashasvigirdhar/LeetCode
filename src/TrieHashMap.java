import java.util.HashMap;

public class TrieHashMap {

  private TrieNode root;

  public TrieHashMap() {
    root = new TrieNode();
  }

  public boolean search(String word) {
    TrieNode t = root;
    for (int i = 0; i < word.length(); i++) {
      if (t.arr.get(word.charAt(i)) == null) {
        return false;
      }
      t = t.arr.get(word.charAt(i));
    }
    return t.isEndOfWord;
  }

  public void insert(String word) {
    TrieNode t = root;
    for (int i = 0; i < word.length(); i++) {
      if (t.arr.get(word.charAt(i)) == null) {
        t.arr.put(word.charAt(i), new TrieNode());
      }
      t = t.arr.get(word.charAt(i));
      t.numWords++;
    }
    t.isEndOfWord = true;
  }


  String findLongestPrefix(String word) {
    TrieNode t = root;
    String ans = null;
    for (int i = 0; i < word.length(); i++) {
      if (t.isEndOfWord) {
        ans = word.substring(0, i);
      }
      t = t.arr.get(word.charAt(i));
      if (t == null) {
        break;
      }
    }
    if (t != null && t.isEndOfWord) {
      ans = word;
    }
    return ans;
  }

  class TrieNode {
    HashMap<Character, TrieNode> arr;
    int numWords;
    boolean isEndOfWord = false;

    // Initialize your data structure here.
    public TrieNode() {
      this.arr = new HashMap<>();
      this.numWords = 0;
    }

  }
}
