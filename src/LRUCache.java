import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

class LRUCache {

  private Map<Integer, Node> map;
  private PriorityQueue<Node> q;
  private int curPVal;
  private int capacity;

  public LRUCache(int capacity) {
    map = new HashMap<>();
    q = new PriorityQueue<>(Comparator.comparingInt(o -> o.pkey));
    curPVal = 0;
    this.capacity = capacity;
  }

  public int get(int key) {
    if (map.containsKey(key)) {
      incrementPVal();
      Node node = map.get(key);
      q.remove(node);
      node.pkey = curPVal;
      q.add(node);
      return node.val;
    }
    return -1;
  }

  private void incrementPVal() {
    curPVal++;
  }

  public void put(int key, int value) {
    incrementPVal();
    Node node;
    if (map.containsKey(key)) {
      node = map.get(key);
      q.remove(node);
      node.val = value;
      node.pkey = curPVal;
      q.add(node);
    } else {
      if (q.size() == capacity) {
        Node poll = q.poll();
        map.remove(poll.key);
      }
      node = new Node(key, value, curPVal);
      q.add(node);
      map.put(key, node);
    }
  }

  class Node {
    int key;
    int val;
    int pkey;

    Node(int key, int val, int pkey) {
      this.key = key;
      this.val = val;
      this.pkey = pkey;
    }
  }
}