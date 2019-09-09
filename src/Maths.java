import java.util.ArrayList;
import java.util.List;

public class Maths {

  public double myPow(double x, int n) {
//    return myPowUtil(x, n);

    if (n < 0) {
      if (n == Integer.MIN_VALUE) {
        return x * myPow(x, n + 1);
      }
      return myPow(1 / x, -n);
    }
    if (n == 0) {
      return 1;
    }
    if (n == 1) {
      return x;
    }
    if (x == 1) {
      return x;
    }
    double ret = myPow(x, n / 2);
    if (n % 2 == 0) {
      return ret * ret;
    } else {
      return ret * ret * x;
    }
  }

  private double myPowUtil(double x, long n) {
    if (n < 0) {
      return myPowUtil(1 / x, -n);
    }
    if (n == 0) {
      return 1;
    }
    if (n == 1) {
      return x;
    }
    if (x == 1) {
      return x;
    }
    double ret = myPowUtil(x, n / 2);
    if (n % 2 == 0) {
      return ret * ret;
    } else {
      return ret * ret * x;
    }
  }

  public boolean isArmstrong(int n) {
    if (n == 0) {
      return true;
    }
    int original = n;
    List<Integer> digits = new ArrayList<>();
    while (n > 0) {
      digits.add(n % 10);
      n /= 10;
    }
    int k = digits.size();
    long sum = 0;
    for (int d : digits) {
      sum += Math.pow(d, k);
    }
    return sum == original;
  }

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
