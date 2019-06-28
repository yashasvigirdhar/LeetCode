class StreamChecker {

  StringBuilder builder;
  private final Trie trie;

  public StreamChecker(String[] words) {
    trie = new Trie();
    for (String word : words) {
      trie.insert(new StringBuilder().append(word).reverse().toString());
    }
    builder = new StringBuilder();
  }

  public boolean query(char letter) {
    builder.insert(0, letter);
    return trie.doesAnyPrefixExists(builder.toString());
  }
}