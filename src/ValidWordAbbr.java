import javax.print.attribute.HashPrintJobAttributeSet;
import java.util.*;

class ValidWordAbbr {

  Map<String, String> map;
  HashSet<String> dict;

  public ValidWordAbbr(String[] dictionary) {
    map = new HashMap<>();
    dict = new HashSet<>();
    StringBuilder sb = new StringBuilder();
    for (String s : dictionary) {
      if (s.length() <= 2) {
        continue;
      }
      dict.add(s);
      sb.setLength(0);
      String key = sb.append(s.charAt(0)).append(s.length() - 2).append(s.charAt(s.length() - 1)).toString();
      if (!map.containsKey(key)) {
        map.put(key, s);
      } else {
        map.put(key, "");
      }
    }
  }

  public boolean isUnique(String s) {
    if (s.length() <= 2) {
      return true;
    }
    StringBuilder sb = new StringBuilder();
    sb.setLength(0);
    String key = sb.append(s.charAt(0)).append(s.length() - 2).append(s.charAt(s.length() - 1)).toString();
    if (map.containsKey(key)) {
      return map.get(key).equals(s);
    }
    return true;
  }
}

/**
 * Your ValidWordAbbr object will be instantiated and called as such:
 * ValidWordAbbr obj = new ValidWordAbbr(dictionary);
 * boolean param_1 = obj.isUnique(word);
 */