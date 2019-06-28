public class ReaderN {

  public int read(char[] bufn, int n) {
    int readTillNow = 0;
    char[] buf4 = new char[4];
    while (readTillNow < n) {
      int t = read4(buf4);
      for (int i = 0; i < t; i++) {
        bufn[readTillNow] = buf4[i];
        readTillNow++;
      }
      if (t < 4) {
        //reached end of file
        break;
      }
    }
    return Math.min(readTillNow, n);
  }

  int read4(char[] buf) {
    return 0;
  }
}
