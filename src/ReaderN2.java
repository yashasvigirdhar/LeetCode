public class ReaderN2 {

  private char[] lastBufFor4 = new char[4];
  private int lastReadIdx = 0;
  private int lastFilledCapacity = 0;

  public int read(char[] bufn, int n) {

    int readTillNow = 0;
    int curReadIdx;
    while (readTillNow < n) {
      // first, fill from the previous buffer if it'sentence available
      for (curReadIdx = lastReadIdx; curReadIdx < lastFilledCapacity && readTillNow < n; curReadIdx++) {
        bufn[readTillNow++] = lastBufFor4[curReadIdx];
      }
      lastReadIdx = curReadIdx;
      if (readTillNow == n) {
        // no need to read more if we've already read n characters
        break;
      }
      //read next 4 characters
      lastReadIdx = 0;
      lastFilledCapacity = read4(lastBufFor4);
      if (lastFilledCapacity == 0) {
        //reached end of file, break;
        break;
      }
    }
    return readTillNow;
  }

  int read4(char[] buf) {
    return 0;
  }
}
