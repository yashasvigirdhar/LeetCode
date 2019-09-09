import java.util.*;

class LogSystem {

  String min, max;
  Map<String, Integer> indices;
  TreeMap<String, Integer> map;

  public LogSystem() {
    min = "2000:00:00:00:00:00";
    max = "2017:12:31:23:60:60";
    indices = new HashMap<>();
    indices.put("Year", 4);
    indices.put("Month", 7);
    indices.put("Day", 10);
    indices.put("Hour", 13);
    indices.put("Minute", 16);
    indices.put("Second", 19);
    map = new TreeMap<>();
  }

  public void put(int id, String timestamp) {
    map.put(timestamp, id);
  }

  public List<Integer> retrieve(String s, String e, String gra) {
    String start = s.substring(0, indices.get(gra)) + min.substring(indices.get(gra));
    String end = e.substring(0, indices.get(gra)) + max.substring(indices.get(gra));
    NavigableMap<String, Integer> subMap = map.subMap(start, true, end, true);
    return new ArrayList<>(subMap.values());
  }
}

/**
 * Your LogSystem object will be instantiated and called as such:
 * LogSystem obj = new LogSystem();
 * obj.put(id,timestamp);
 * List<Integer> param_2 = obj.retrieve(s,e,gra);
 */