public class WordDictionary {

  private TrieII trieII;

  public WordDictionary() {
    trieII = new TrieII();
  }

  /**
   * Adds a word into the data structure.
   */
  public void addWord(String word) {
    trieII.addWord(word);
  }

  /**
   * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
   */
  public boolean search(String word) {
    return trieII.isWordPresent(word);
  }

  class TrieII {
    TrieNodeII root;

    public TrieII() {
      root = new TrieNodeII();
    }

    public void addWord(String word) {
      addWord(root, word);
    }

    private void addWord(TrieNodeII root, String word) {
      while (word.length() > 0) {
        int idx = word.charAt(0) - 'a';
        if (root.children[idx] == null) {
          root.children[idx] = new TrieNodeII();
        }
        root = root.children[idx];
        word = word.substring(1);
      }
      root.wordsEndingHere++;
    }

    public boolean isWordPresent(String word) {
      return isWordPresent(root, word);
    }

    private boolean isWordPresent(TrieNodeII root, String word) {
      if (word.length() == 0) {
        return root.wordsEndingHere > 0;
      }
      char c = word.charAt(0);
      String substring = word.substring(1);
      if (c == '.') {
        for (int i = 0; i < root.children.length; i++) {
          if (root.children[i] != null && isWordPresent(root.children[i], substring)) {
            return true;
          }
        }
        return false;
      } else {
        return (root.children[c - 'a'] != null)
                && isWordPresent(root.children[c - 'a'], substring);
      }
    }
  }

  class TrieNodeII {
    TrieNodeII[] children = new TrieNodeII[26];
    int wordsEndingHere;
  }
}



