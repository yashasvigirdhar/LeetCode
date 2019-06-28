import java.util.ArrayList;
import java.util.List;

public class CodecString {

  private static final String DELIM = "|";

  public String encode(List<String> strs) {
    StringBuilder b = new StringBuilder();
    for (String s : strs) {
      b.append(s.length()).append(DELIM).append(s);
    }
    return b.toString();
  }

  // Decodes a single string to a list of strings.
  public List<String> decode(String s) {
    List<String> res = new ArrayList<>();
    int idx = 0;
    while (idx < s.length()) {
      int b = idx;
      while (s.charAt(idx) >= '0' && s.charAt(idx) <= '9') {
        idx++;
      }
      int len = Integer.parseInt(s.substring(b, idx));
      res.add(s.substring(idx + 1, idx + len + 1));
      idx = idx + len + 1;
    }
    return res;
  }
}
