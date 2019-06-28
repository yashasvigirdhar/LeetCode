public class Maths {

  public int divide(int dividend, int divisor) {
    int absDivisor = Math.abs(divisor);
    int res = 1;
    if (dividend < 0) {
      while (dividend + absDivisor <= 0) {
        dividend += absDivisor;
        if (res == Integer.MAX_VALUE) {
          if (divisor < 0) {
            return res;
          } else {
            return res * -1 - 1;
          }
        }
        res++;
      }
      if (divisor > 0) {
        return res * -1;
      }
      return res;
    } else {
      while (dividend - absDivisor >= 0) {
        dividend -= absDivisor;
        if (res == Integer.MAX_VALUE) {
          if (divisor < 0) {
            return res * -1;
          } else {
            return res;
          }
        }
        res++;
      }
      if (divisor > 0) {
        return res;
      }
      return res * -1;
    }
  }
}
