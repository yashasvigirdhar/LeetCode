import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Codec3 {

  Map<String, String> map;
  char[] chars = new char[]{
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
  };
  int len = 6;

  Codec3() {
    map = new HashMap<>();
  }

  // Encodes a URL to a shortened URL.
  public String encode(String longUrl) {
    String str = generate();
    while (map.containsKey(str)) {
      str = generate();
    }
    map.put(str, longUrl);
    return str;
  }

  // Decodes a shortened URL to its original URL.
  public String decode(String shortUrl) {
    return map.get(shortUrl);
  }

  private String generate() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < len; i++) {
      sb.append(chars[new Random().nextInt(chars.length)]);
    }
    return sb.toString();
  }

}
