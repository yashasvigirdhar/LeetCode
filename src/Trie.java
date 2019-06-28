public class Trie {

  private TrieNode root;

  public Trie() {
    root = new TrieNode();
  }

  void print() {
    TrieNode p = root;
    for (int i = 0; i < 26; i++) {
      if (p.arr[i] != null) {
        p = p.arr[i];
        System.out.println((char) i + 'a');
      }
    }
  }

  public boolean search(String word) {
    TrieNode t = root;
    for (int i = 0; i < word.length(); i++) {
      if (t.arr[word.charAt(i) - 'a'] == null) {
        return false;
      }
      t = t.arr[word.charAt(i) - 'a'];
    }
    return t.isEndOfWord;
  }

  public void insert(String word) {
    TrieNode t = root;
    for (int i = 0; i < word.length(); i++) {
      if (t.isEndOfWord) {
        return;
      }
      int idx = word.charAt(i) - 'a';
      if (t.arr[idx] == null) {
        t.arr[idx] = new TrieNode();
      }
      t = t.arr[idx];
      t.numWords++;
    }
    t.isEndOfWord = true;
  }

  /**
   * Returns if there is any word in the trie that starts with the given prefix.
   */
  public boolean startsWith(String prefix) {
    TrieNode t = root;
    for (int i = 0; i < prefix.length(); i++) {
      if (t.arr[prefix.charAt(i) - 'a'] == null) {
        return false;
      }
    }
    return true;
  }

  String findLongestPrefix(String word) {
    TrieNode t = root;
    String ans = null;
    for (int i = 0; i < word.length(); i++) {
      if (t.isEndOfWord) {
        ans = word.substring(0, i);
      }
      t = t.arr[word.charAt(i) - 'a'];
      if (t == null) {
        break;
      }
    }
    if (t.isEndOfWord) {
      ans = word;
    }
    return ans;
  }

  boolean doesAnyPrefixExists(String word) {
    TrieNode t = root;
    for (int i = 0; i < word.length(); i++) {
      if (t.isEndOfWord) {
        return true;
      }
      t = t.arr[word.charAt(i) - 'a'];
      if (t == null) {
        return false;
      }
    }
    return t.isEndOfWord;
  }


  String findUniquePrefix(String word) {
    TrieNode t = root;
    for (int i = 0; i < word.length(); i++) {
      if (t.isEndOfWord) {
        return word.substring(0, i);
      }
      t = t.arr[word.charAt(i) - 'a'];
      if (t == null) {
        return word;
      }
    }
    return word;
  }

  class TrieNode {
    TrieNode[] arr;
    int numWords;
    boolean isEndOfWord = false;

    // Initialize your data structure here.
    public TrieNode() {
      this.arr = new TrieNode[26];
      this.numWords = 0;
    }

  }
}
