public class TestJava {

  static int b2 = Inner.c;
  static int b = new Inner().getN();

  static class Inner {
    int a = b;
    static int c = 0;

    int getN() {
      c++;
      return c;
    }
  }
}
