class StringIterator {

  int count, idx;
  char curChar;
  String str;

  public StringIterator(String compressedString) {
    str = compressedString;
    initializeNextNumber();
  }

  private void initializeNextNumber() {
    if (idx == str.length()) {
      return;
    }
    curChar = str.charAt(idx++);
    StringBuilder t = new StringBuilder();
    while (idx < str.length() && Character.isDigit(str.charAt(idx))) {
      t.append(str.charAt(idx++));
    }
    count = Integer.parseInt(t.toString());
  }

  public char next() {
    if (count == 0) {
      initializeNextNumber();
    }
    if (count > 0) {
      count--;
      return curChar;
    } else {
      return ' ';
    }
  }

  public boolean hasNext() {
    if (count > 0) {
      return true;
    }
    return idx == str.length();
  }
}

/**
 * Your StringIterator object will be instantiated and called as such:
 * StringIterator obj = new StringIterator(compressedString);
 * char param_1 = obj.next();
 * boolean param_2 = obj.hasNext();
 */