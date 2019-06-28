import java.util.ArrayList;
import java.util.List;

public class BitManipulation {

  public List<Boolean> prefixesDivBy5(int[] A) {
    int num = 0;
    List<Boolean> res = new ArrayList<>();
    for (int aA : A) {
      num = num * 2;
      if (aA == 1) {
        num += 1;
      }
      num %= 5;
      if (num == 0) {
        res.add(true);
      } else {
        res.add(false);
      }
    }
    return res;
  }
}
