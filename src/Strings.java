import java.util.*;

public class Strings {

  public String mostCommonWord(String paragraph, String[] banned) {
    HashSet<String> ban = new HashSet<>(Arrays.asList(banned));
    int maxCount = 0;
    String ans = "";
    Map<String, Integer> counts = new HashMap<>();
    char[] chars = paragraph.toCharArray();
    StringBuilder curBuilder = new StringBuilder();
    for (char c : chars) {
      if (Character.isLetter(c)) {
        curBuilder.append(Character.toLowerCase(c));
      } else {
        if (curBuilder.length() > 0) {
          String word = curBuilder.toString();
          if (!ban.contains(word)) {
            counts.put(word, counts.getOrDefault(word, 0) + 1);
            if (counts.get(word) > maxCount) {
              maxCount = counts.get(word);
              ans = word;
            }
          }
        }
        curBuilder = new StringBuilder();
      }
    }
    if (curBuilder.length() > 0) {
      String word = curBuilder.toString();
      if (!ban.contains(word)) {
        counts.put(word, counts.getOrDefault(word, 0) + 1);
        if (counts.get(word) > maxCount) {
          maxCount = counts.get(word);
          ans = word;
        }
      }
    }
    return ans;
  }

  public String[] reorderLogFiles(String[] logs) {
    List<String> numLogs = new ArrayList<>();
    TreeMap<String, String> strLogs = new TreeMap<>();
    for (String log : logs) {
      String[] words = log.split(" ");
      if (words[1].charAt(0) >= 'a' && words[1].charAt(0) <= 'z') {
        StringBuilder b = new StringBuilder();
        for (int i = 1; i < words.length; i++) {
          b.append(words[i]).append('#');
        }
        b.append(words[0]);
        strLogs.put(b.toString(), log);
      } else {
        numLogs.add(log);
      }
    }
    String[] ans = new String[logs.length];
    int idx = 0;
    for (Map.Entry<String, String> entry : strLogs.entrySet()) {
      ans[idx++] = entry.getValue();
    }
    for (String numLog : numLogs) {
      ans[idx++] = numLog;
    }
    return ans;
  }

  public boolean rotateString(String A, String B) {
    if (A.length() != B.length()) {
      return false;
    }
    if (A.length() == 0) {
      return true;
    }
    char c = A.charAt(0);
    List<Integer> startingIndicesInB = new ArrayList<>();
    for (int i = 0; i < B.length(); i++) {
      if (B.charAt(i) == c) {
        startingIndicesInB.add(i);
      }
    }
    for (int startIdx : startingIndicesInB) {
      int i = 0, j = startIdx;
      int curCount = 0;
      while (curCount < A.length()) {
        if (A.charAt(i) != B.charAt(j)) {
          break;
        }
        curCount++;
        i++;
        j++;
        if (j == B.length()) {
          j = 0;
        }
      }
      if (curCount == A.length()) {
        return true;
      }
    }
    return false;
  }


  public int[] diStringMatchIII(String s) {
    int totalDLeft = 0;
    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) == 'D') {
        totalDLeft++;
      }
    }
    int[] ans = new int[s.length() + 1];
    ans[0] = totalDLeft;
    int nextI = totalDLeft + 1;
    int nextD = totalDLeft - 1;
    int idx = 0;
    while (idx < s.length()) {
      if (s.charAt(idx) == 'I') {
        ans[idx + 1] = nextI++;
      } else {
        ans[idx + 1] = nextD--;
      }
      idx++;
    }
    return ans;
  }

  public int[] diStringMatchII(String s) {
    boolean[] chosen = new boolean[s.length() + 1];
    Arrays.fill(chosen, false);
    int totalDLeft = 0;
    for (int i = 0; i < s.length(); i++)
      if (s.charAt(i) == 'D') {
        totalDLeft++;
      }
    int[] ans = new int[s.length() + 1];
    int idx = -1;
    while (idx < s.length()) {
      if (idx != -1 && s.charAt(idx) == 'D') {
        totalDLeft--;
      }
      int dCount = 0, newElement = -1;
      int i = 0;
      while (i < chosen.length) {
        if (!chosen[i]) {
          if (dCount < totalDLeft) {
            dCount++;
          } else {
            // it will always land here since there always exists a solution.
            newElement = i;
            break;
          }
        }
        i++;
      }
      chosen[newElement] = true;
      ans[idx + 1] = newElement;

      idx++;
    }
    return ans;
  }


  public int[] diStringMatch(String s) {
    boolean[] chosen = new boolean[s.length() + 1];
    Arrays.fill(chosen, false);
    for (int i = 0; i < chosen.length; i++) {
      chosen[i] = true;
      List<Integer> returnedList = diStringMatchUtil(0, chosen, s, i);
      if (returnedList != null) {
        returnedList.add(0, i);
        return returnedList.stream().mapToInt(Integer::intValue).toArray();
      }
      chosen[i] = false;
    }
    return null;
  }

  private List<Integer> diStringMatchUtil(int idx, boolean[] chosen, String s, int prev) {
    if (idx == s.length()) {
      // update global string
      return new ArrayList<>();
    }
    char curChar = s.charAt(idx);

    for (int i = 0; i < chosen.length; i++) {
      if (!chosen[i]) {
        if ((curChar == 'I' && i > prev) || (curChar == 'D' && i < prev)) {
          chosen[i] = true;
          List<Integer> returnedList = diStringMatchUtil(idx + 1, chosen, s, i);
          if (returnedList != null) {
            returnedList.add(0, i);
            return returnedList;
          }
          chosen[i] = false;
        }
      }
    }
    return null;
  }


  public int uniqueLetterStringIII(String s) {
    int idx = 0;
    int[] contri = new int[26];
    int[] lastPos = new int[26];
    int result = 0;
    while (idx < s.length()) {
      int curChar = s.charAt(idx) - 'A';
      contri[curChar] = idx + 1 - lastPos[curChar];
      int cur = 0;
      lastPos[curChar] = idx + 1;
      for (int i = 0; i < 26; i++) {
        cur += contri[i];
      }
      result += cur;
    }
    return result;


  }

  public int uniqueLetterStringII(String s) {
    int[] lastPosition = new int[26];
    int[] contribution = new int[26];
    int res = 0;

//     Basically, at each it, we count the contribution of all the characters to all the substrings ending till that point.

    for (int i = 0; i < s.length(); i++) {

      int curChar = s.charAt(i) - 'A';

//       Now, we need to update the contribution of curChar.
//       The total number of substrings ending at i are i+1. So if it was a unique character, it'd contribute to all of those
//       and it's contribution would have been i+1.
//       But if it's repeating, it means it has already contributed previously. So remove it's previous contribution.
//       We can do that as we have it's last position.
//       So these are the contributions for strings which start after this character's last occurrence and end at i.
//       A simple example will demonstrate that the number of these strings are i+1 - lastPosition[curChar]
//       For characters not appeared till now, lastPosition[curChar] would be 0.
      int totalNumOfSubstringsEndingHere = i + 1;
      contribution[curChar] = totalNumOfSubstringsEndingHere - lastPosition[curChar];

//       Note that, the contribution of all the other characters will remain same.

//       count the cur answer by summing all the contributions. This loop can be avoided by the idea in original post, but I find
//       it easy to understand with this and it only iterates over 26 values.
      int cur = 0;
      for (int j = 0; j < 26; j++) {
        cur += contribution[j];
      }

//       add the current value to final answer.
      res += cur;

//       update the last position of this char. This helps in future to count it's contribution if it appears again.
      lastPosition[curChar] = i + 1;
    }
    return res;
  }

  public int uniqueLetterString(String s) {
    int[] map = new int[26];
    int count = 0;
    int mod = 1000000007;
    for (int i = 0; i < s.length(); i++) {
      Arrays.fill(map, 0);
      int curUnique = 0;
      int j = i;
      while (j < s.length()) {
        map[s.charAt(j) - 'A']++;
        if (map[s.charAt(j) - 'A'] == 1) {
          curUnique++;
        } else if (map[s.charAt(j) - 'A'] == 2) {
          curUnique--;
        }
        count = (curUnique % mod + curUnique) % mod;
        j++;
      }
    }
    return count;
  }

  public boolean backspaceCompare(String S, String T) {
    S = processBackspace(S);
    T = processBackspace(T);
    if (S.length() != T.length()) {
      return false;
    }
    int i = 0;
    while (i < S.length()) {
      if (S.charAt(i) != T.charAt(i)) {
        return false;
      }
      i++;
    }
    return true;
  }

  private String processBackspace(String s) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) == '#') {
        if (sb.length() > 0) {
          sb.deleteCharAt(sb.length() - 1);
        }
      } else {
        sb.append(s.charAt(i));
      }
    }
    return sb.toString();
  }

  public String addBoldTagII(String s, String[] dict) {
    TrieHashMap trie = new TrieHashMap();
    for (String word : dict) {
      trie.insert(word);
    }
    boolean[] bold = new boolean[s.length()];
    Arrays.fill(bold, false);
    int curStart;
    int idx = 0;
    while (idx < s.length()) {
      String curStr = s.substring(idx);
      // find longest prefix of curStr which exists in dict.
      String longestPrefix = trie.findLongestPrefix(curStr);
      if (longestPrefix != null) {
        curStart = idx;
        int curEnd = longestPrefix.length() + idx - 1;
        for (int i = curStart; i <= curEnd && i < s.length(); i++) {
          bold[i] = true;
        }
      }
      idx++;
    }
    StringBuilder builder = new StringBuilder();
    int start = 0;
    while (start < s.length()) {
      if (!bold[start]) {
        builder.append(s.charAt(start));
        start++;
        continue;
      }
      int end = start;
      while (end < s.length() && bold[end]) {
        end++;
      }
      builder.append("<b>");
      builder.append(s, start, end);
      builder.append("</b>");
      start = end;
    }
    return builder.toString();
  }

  public String addBoldTag(String s, String[] dict) {
    TrieHashMap trie = new TrieHashMap();
    for (String word : dict) {
      trie.insert(word);
    }
    List<Integer> indices = new ArrayList<>();
    int curStart;
    int idx = 0;
    while (idx < s.length()) {
      String curStr = s.substring(idx);
      // find longest prefix of curStr which exists in dict.
      String longestPrefix = trie.findLongestPrefix(curStr);
      if (longestPrefix != null) {
        curStart = idx;
        int curEnd = longestPrefix.length() + idx - 1;
        for (int i = idx + 1; i <= curEnd + 1 && i < s.length(); i++) {
          curStr = s.substring(i);
          longestPrefix = trie.findLongestPrefix(curStr);
          if (longestPrefix != null) {
            curEnd = Math.max(curEnd, longestPrefix.length() + i - 1);
          }
        }
        indices.add(curStart);
        indices.add(curEnd);
        idx = curEnd + 2;
      } else {
        idx++;
      }
    }
    idx = 0;
    StringBuilder builder = new StringBuilder();
    int prevStart = 0;
    while (idx < indices.size()) {
      int start = indices.get(idx++);
      int end = indices.get(idx++);
      builder.append(s, prevStart, start);
      builder.append("<b>");
      builder.append(s, start, end + 1);
      builder.append("</b>");
      prevStart = end + 1;
    }
    builder.append(s, prevStart, s.length());
    return builder.toString();
  }

  public List<Integer> partitionLabels(String S) {
    int[] indices = new int[26];
    Arrays.fill(indices, -1);
    for (int i = 0; i < S.length(); i++) {
      indices[S.charAt(i) - 'a'] = i;
    }
    List<Integer> ans = new ArrayList<>();
    int curStart = -1, curEnd = -1;
    for (int i = 0; i < S.length(); i++) {
      if (curStart == -1) {
        curStart = i;
      }
      if (indices[S.charAt(i) - 'a'] != -1) {
        curEnd = Math.max(curEnd, indices[S.charAt(i) - 'a']);
      }
      if (curEnd == S.length() - 1) {
        ans.add(curEnd - curStart + 1);
        break;
      }
      if (curEnd == i) {
        ans.add(curEnd - curStart + 1);
        curStart = -1;
        curEnd = -1;
      }
    }
    return ans;
  }

  public boolean isStrobogrammatic(String num) {
    StringBuilder builder = new StringBuilder(num.length());
    for (int i = num.length() - 1; i >= 0; i--) {
      switch (num.charAt(i)) {
        case '0':
          builder.append('0');
          break;
        case '1':
          builder.append('1');
          break;
        case '6':
          builder.append('9');
          break;
        case '8':
          builder.append('8');
          break;
        case '9':
          builder.append('6');
          break;
        default:
          return false;
      }
    }
    return num.equals(builder.toString());
  }

  public int expressiveWords(String S, String[] words) {
    int ans = 0;
    for (String word : words) {
      if (isStretchy(S, word)) {
        ans++;
      }
    }
    return ans;
  }

  private boolean isStretchy(String query, String word) {
    int i = 0, j = 0;
    while (i < query.length() && j < word.length()) {
      char curChar = query.charAt(i);
      int count1 = 0;
      while (i < query.length() && query.charAt(i) == curChar) {
        count1++;
        i++;
      }
      int count2 = 0;
      while (j < word.length() && word.charAt(j) == curChar) {
        count2++;
        j++;
      }
      if (count2 == 0) {
        return false;
      }
      if (count1 == count2) {
        continue;
      } else if (count1 < count2) {
        return false;
      } else {
        if (count1 < 3) {
          return false;
        }
      }
    }
    return i == query.length() && j == word.length();
  }

  public boolean judgeCircle(String moves) {
    int x = 0, y = 0;
    for (int i = 0; i < moves.length(); i++) {
      switch (moves.charAt(i)) {
        case 'R':
          x++;
          break;
        case 'L':
          x--;
          break;
        case 'U':
          y++;
          break;
        case 'D':
          y--;
          break;
      }
    }
    return (x == 0 && y == 0);
  }

  public int numUniqueEmails(String[] emails) {
    HashSet<String> set = new HashSet<>();
    for (String email : emails) {
      StringBuilder builder = new StringBuilder();
      int idx = 0;
      boolean domainStarted = false;
      boolean ignoreBeforeDomain = false;
      while (idx < email.length()) {
        char curChar = email.charAt(idx);
        if (curChar == '@') {
          domainStarted = true;
          builder.append(curChar);
          idx++;
          continue;
        }
        if (domainStarted) {
          builder.append(curChar);
          idx++;
          continue;
        }
        if (curChar == '.') {
          idx++;
          continue;
        }
        if (curChar == '+') {
          ignoreBeforeDomain = true;
          idx++;
          continue;
        }
        if (!ignoreBeforeDomain) {
          builder.append(curChar);
        }
        idx++;
      }
      set.add(builder.toString());
    }
    return set.size();
  }


  public List<Boolean> camelMatch(String[] queries, String pattern) {
    List<Boolean> result = new ArrayList<>();
    for (String s : queries) {
      result.add(camelMatch(s, pattern));
    }
    return result;
  }

  private boolean camelMatch(String query, String pattern) {
    int i = 0, j = 0;
    while (i < query.length() && j < pattern.length()) {
      if (query.charAt(i) == pattern.charAt(j)) {
        i++;
        j++;
        continue;
      }
      if (isLowercaseAlpha(query, i)) {
        i++;
        continue;
      }
      return false;
    }
    if (j < pattern.length()) {
      return false;
    }
    while (i < query.length()) {
      if (!isLowercaseAlpha(query, i)) {
        return false;
      }
      i++;
    }
    return true;
  }

  private boolean isLowercaseAlpha(String query, int i) {
    return query.charAt(i) >= 'a' && query.charAt(i) <= 'z';
  }

  public String removeOuterParentheses(String S) {
    int ctr = -1;
    int prev = -1;
    char[] c = S.toCharArray();
    for (int i = 0; i < c.length; i++) {
      if (c[i] == '(') {
        ctr++;
        if (ctr == 1) {
          prev = i;
        }
      } else {
        ctr--;
      }
      if (ctr == 0) {
        c[prev] = '#';
        c[i] = '#';
      }
    }
    StringBuilder builder = new StringBuilder();
    for (char aC : c) {
      if (aC != '#') {
        builder.append(aC);
      }
    }
    return builder.toString();
  }

  public String replaceWords(List<String> dict, String sentence) {
    Trie trie = new Trie();
    for (String s : dict) {
      trie.insert(s);
    }
    String[] words = sentence.split(" ");
    for (int i = 0; i < words.length; i++) {
      words[i] = trie.findUniquePrefix(words[i]);
    }
    StringBuilder builder = new StringBuilder();
    for (String word : words) {
      builder.append(word).append(" ");
    }
    return builder.substring(0, builder.length() - 1);
  }

  public List<Integer> findAnagrams(String s, String p) {
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < p.length(); i++) {
      map.put(p.charAt(i), map.getOrDefault(p.charAt(i), 0) + 1);
    }
    List<Integer> ans = new ArrayList<>();
    int l = 0, r = 0;
    Map<Character, Integer> map2 = new HashMap<>();
    int formed = 0;
    while (r < s.length()) {
      char curChar = s.charAt(r);
      map2.put(curChar, map2.getOrDefault(curChar, 0) + 1);
      Integer requiredCount = map.getOrDefault(curChar, 0);
      Integer curCount = map2.getOrDefault(curChar, 0);
      if (requiredCount.equals(curCount)) {
        formed++;
      }
      while (l <= r && formed == map.size()) {
        if (map.getOrDefault(s.charAt(l), 0).equals(map2.get(s.charAt(l)))) {
          formed--;
        }
        if (s.charAt(l) == curChar) {
          curCount--;
        }
        map2.put(s.charAt(l), map2.get(s.charAt(l)) - 1);
        if (map2.get(s.charAt(l)) == 0) {
          map2.remove(s.charAt(l));
        }
        l++;
      }
      if (compareMaps(map, map2) == 0) {
        ans.add(l);
      }
      r++;
    }
    return ans;
  }

  private int compareMaps(Map<Character, Integer> map, Map<Character, Integer> map2) {
    if (map2.size() > map.size()) return 1;
    for (Map.Entry<Character, Integer> entry : map.entrySet()) {
      if (map2.getOrDefault(entry.getKey(), 0) > entry.getValue()) {
        return 1;
      } else if (map2.getOrDefault(entry.getKey(), 0) < entry.getValue()) {
        return -1;
      }
    }
    return 0;
  }

  public boolean isAnagram(String s, String t) {
    int[] counter = new int[26];
    Arrays.fill(counter, 0);
    for (int i = 0; i < s.length(); i++) {
      counter[s.charAt(i) - 'a']++;
    }
    for (int i = 0; i < t.length(); i++) {
      counter[t.charAt(i) - 'a']--;
    }
    for (int i = 0; i < 26; i++) {
      if (counter[i] != 0) return false;
    }
    return true;
  }

  private String encode(String str) {
    int[] counter = new int[26];
    Arrays.fill(counter, 0);
    for (int i = 0; i < str.length(); i++) {
      counter[str.charAt(i) - 'a']++;
    }
    StringBuilder keyBuilder = new StringBuilder();
    for (int i = 0; i < 26; i++) {
      if (counter[i] > 0) {
        keyBuilder.append(counter[i]).append('a' + i);
      }
    }
    return keyBuilder.toString();
  }

  public List<List<String>> groupAnagrams(String[] strs) {
    HashMap<String, List<String>> map = new HashMap<>();
    for (String str : strs) {
      int[] counter = new int[26];
      Arrays.fill(counter, 0);
      for (int i = 0; i < str.length(); i++) {
        counter[str.charAt(i) - 'a']++;
      }
      StringBuilder keyBuilder = new StringBuilder();
      for (int i = 0; i < 26; i++) {
        if (counter[i] > 0) {
          keyBuilder.append(counter[i]).append('a' + i);
        }
      }
      String key = keyBuilder.toString();
      if (!map.containsKey(key)) {
        map.put(key, new ArrayList<>());
      }
      map.get(key).add(str);
    }
    return new ArrayList<>(map.values());
  }

  public boolean isOneEditDistance(String s, String t) {
    for (int i = 0; i < Math.min(s.length(), t.length()); i++) {
      if (s.charAt(i) != t.charAt(i)) {
        if (s.length() == t.length()) {
          return s.substring(i + 1).equals(t.substring(i + 1));
        } else if (s.length() < t.length()) {
          return s.substring(i).equals(t.substring(i + 1));
        } else {
          return s.substring(i + 1).equals(t.substring(i));
        }
      }
    }
    return Math.abs(s.length() - t.length()) == 1;
  }

  public String findLongestWord(String s, List<String> d) {
    List<String> candidates = new ArrayList<>();
    int maxCount = 0;
    for (String str : d) {
      if (canAchieve(s, str)) {
        if (str.length() >= maxCount) {
          maxCount = str.length();
          candidates.add(str);
        }
      }
    }
    candidates.sort(String::compareTo);
    for (String str : candidates) {
      if (str.length() == maxCount) {
        return str;
      }
    }
    return "";
  }

  private boolean canAchieve(String s1, String s2) {
    int i = 0, j = 0;
    while (i < s1.length()) {
      if (s1.charAt(i) == s2.charAt(j)) {
        j++;
      }
      if (j == s2.length()) {
        return true;
      }
      i++;
    }
    return false;
  }

  public String reverseWords(String s) {
    char[] chars = s.toCharArray();
    int n = chars.length;
    int lastStart = -1;
    int i = 0;
    while (i < n) {
      if (chars[i] == ' ') {
        if (lastStart == -1) {
          continue;
        } else {
          reverseString(chars, lastStart, i - 1);
          lastStart = -1;
        }
      } else {
        if (lastStart == -1) {
          lastStart = i;
        }
      }
      i++;
    }
    if (lastStart != -1) {
      reverseString(chars, lastStart, n - 1);
    }
    return new String(chars);
  }

  public String reverseStr2(String s, int k) {
    char[] chars = s.toCharArray();
    int n = chars.length;
    if (n == 0) {
      return s;
    }
    int i = 0;
    while (i < n) {
      int j = Math.min(i + k - 1, n - 1);
      reverseString(chars, i, j);
      i += 2 * k;
    }
    return new String(chars);
  }

  public String reverseStr(String s, int k) {
    char[] chars = s.toCharArray();
    int n = chars.length;
    if (n == 0) {
      return s;
    }
    int i = 0;
    int curCount = 1, start = 0;
    while (i < n) {
      if (curCount == 2 * k) {
        reverseString(chars, start, start + k - 1);
        start = i + 1;
        curCount = 0;
      }
      i++;
      curCount++;
    }
    if (curCount > k) {
      reverseString(chars, start, start + k - 1);
    } else if (curCount > 0) {
      reverseString(chars, start, start + curCount - 2);
    }
    return new String(chars);
  }

  public void reverseString(char[] s, int b, int e) {
    int n = e - b + 1;
    if (n == 0 || n == 1) return;
    int l = b, r = e;
    while (l <= r) {
      char temp = s[l];
      s[l] = s[r];
      s[r] = temp;
      l++;
      r--;
    }
  }

  public String reverseVowels(String s) {
    if (s == null) {
      return s;
    }
    char[] chars = s.toCharArray();
    int n = chars.length;
    if (n == 0 || n == 1) {
      return s;
    }
    int l = 0, r = n - 1;
    while (l <= r) {

      while (l < r && !isVowel(chars[l])) {
        l++;
      }
      while (l < r && !isVowel(chars[r])) {
        r--;
      }
      char temp = chars[l];
      chars[l] = chars[r];
      chars[r] = temp;
      l++;
      r--;
    }
    return new String(chars);
  }

  private boolean isVowel(char charAt) {
    return charAt == 'a' || charAt == 'e' || charAt == 'i' || charAt == 'o' || charAt == 'u'
            || charAt == 'A' || charAt == 'E' || charAt == 'I' || charAt == 'O' || charAt == 'U';
  }

  public String minWindow(String s, String t) {
    int n = s.length();
    if (n == 0) {
      return "";
    }
    HashMap<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < t.length(); i++) {
      Character c = t.charAt(i);
      if (!map.containsKey(c)) {
        map.put(c, 0);
      }
      map.put(c, map.get(c) + 1);
    }

    int l = 0, r = 0;
    HashMap<Character, Integer> counts = new HashMap<>();
    if (!counts.containsKey(s.charAt(0))) {
      counts.put(s.charAt(0), 0);
    }
    counts.put(s.charAt(0), counts.get(s.charAt(0)) + 1);

    int gl = -1, gr = -1, minCount = Integer.MAX_VALUE;
    boolean flag = true;
    while (flag) {

      while (r + 1 < s.length() && !isPossibleSolution(counts, map)) {
        r++;
        if (!counts.containsKey(s.charAt(r))) {
          counts.put(s.charAt(r), 0);
        }
        counts.put(s.charAt(r), counts.get(s.charAt(r)) + 1);
      }
      if (r + 1 == s.length()) {
        flag = false;
      }

      while (l <= r && isPossibleSolution(counts, map)) {
        int curCount;
        curCount = r - l + 1;
        if (curCount < minCount) {
          minCount = curCount;
          gl = l;
          gr = r;
        }
        counts.put(s.charAt(l), counts.get(s.charAt(l)) - 1);
        l++;
      }
    }
    if (isPossibleSolution(counts, map)) {
      int curCount = r - l + 1;
      if (curCount < minCount) {
        minCount = curCount;
        gl = (l == 0) ? 0 : l - 1;
        gr = r;
      }
    }
    return (minCount == Integer.MAX_VALUE) ? "" : s.substring(gl, gr + 1);
  }

  private boolean isPossibleSolution(HashMap<Character, Integer> count, HashMap<Character, Integer> map) {
    boolean possibleSolution = true;
    for (Map.Entry<Character, Integer> entry : map.entrySet()) {
      if (!count.containsKey(entry.getKey()) || count.get(entry.getKey()) < entry.getValue()) {
        possibleSolution = false;
        break;
      }
    }
    return possibleSolution;
  }
}
