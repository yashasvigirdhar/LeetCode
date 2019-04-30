import java.util.HashMap;
import java.util.Map;

class LRUCache2 {

  private Map<Integer, Node> map;
  private DoublyLinkedList d;
  private int capacity;

  public LRUCache2(int capacity) {
    map = new HashMap<>();
    d = new DoublyLinkedList();
    this.capacity = capacity;
  }

  public int get(int key) {
    if (map.containsKey(key)) {
      Node node = map.get(key);
      d.remove(node);
      d.add(node);
      return node.val;
    }
    return -1;
  }

  public void put(int key, int value) {
    Node node;
    if (map.containsKey(key)) {
      node = map.get(key);
      d.remove(node);
      node.val = value;
      d.add(node);
    } else {
      if (map.size() == capacity) {
        Node removed = d.removeAsPerLRU();
        map.remove(removed.key);
      }
      node = new Node(key, value);
      d.add(node);
      map.put(key, node);
    }
  }

  class DoublyLinkedList {
    Node head;
    Node tail;

    public void makeHead(Node node) {
      node.next = head;
      if (head != null) {
        head.prev = node;
      }
      head = node;

      if (tail == null) {
        tail = node;
      }
    }

    public Node remove(Node node) {
      if (node.prev == null) {
        head = node.next;
      } else {
        node.prev.next = node.next;
      }
      if (node.next == null) {
        tail = node.prev;
      } else {
        node.next.prev = node.prev;
      }
      node.next = null;
      node.prev = null;
      return node;
    }

    public Node removeAsPerLRU() {
      return remove(tail);
    }

    public void add(Node node) {
      makeHead(node);
    }

  }

  class Node {
    int key;
    int val;
    Node next = null;
    Node prev = null;

    Node(int key, int val) {
      this.key = key;
      this.val = val;
    }
  }
}