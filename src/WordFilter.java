class WordFilter {

  MyTrie t;

  public WordFilter(String[] words) {
    t = new MyTrie();
    for (int i = 0; i < words.length; i++) {
      for (int j = 0; j <= words[i].length(); j++) {
        String toInsert = words[i].substring(j) + "{" + words[i];
        t.insert(toInsert, i);
      }
    }
  }

  public int f(String prefix, String suffix) {
    return t.search(suffix + "{" + prefix);
  }

  public class MyTrie {

    private TrieNode root;

    public MyTrie() {
      root = new TrieNode();
    }

    public int search(String word) {
      TrieNode t = root;
      for (int i = 0; i < word.length(); i++) {
        char c = word.charAt(i);
        if (t.arr[c - 'a'] == null) {
          return -1;
        }
        t = t.arr[c - 'a'];
      }
      return t.weight;
    }

    public void insert(String word, int weight) {
      TrieNode t = root;
      for (int i = 0; i < word.length(); i++) {
        int idx = word.charAt(i) - 'a';
        if (t.arr[idx] == null) {
          t.arr[idx] = new TrieNode();
        }
        t = t.arr[idx];
        t.weight = Math.max(t.weight, weight);
      }
      t.weight = weight;
    }

    class TrieNode {
      TrieNode[] arr;
      int weight;

      // Initialize your data structure here.
      public TrieNode() {
        this.arr = new TrieNode[27];
        this.weight = 0;
      }

    }
  }
}


