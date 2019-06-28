class MyHashMap {

  private int BUCKETS = 1000;

  private Node[] values;

  /**
   * Initialize your data structure here.
   */
  public MyHashMap() {
    values = new Node[BUCKETS];
  }

  /**
   * value will always be non-negative.
   */
  public void put(int key, int value) {
    int hashKey = getHashKey(key);
    Node head = values[hashKey];
    if (head == null) {
      values[hashKey] = new Node(key, value);
      return;
    }
    while (head.next != null) {
      if (head.key == key) {
        head.value = value;
        return;
      }
      head = head.next;
    }
    if (head.key == key) {
      head.value = value;
    } else {
      head.next = new Node(key, value);
    }
  }

  private int getHashKey(int key) {
    return java.lang.Integer.hashCode(key) % BUCKETS;
  }

  /**
   * Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
   */
  public int get(int key) {
    Node head = values[getHashKey(key)];
    while (head != null) {
      if (head.key == key) {
        return head.value;
      }
      head = head.next;
    }
    return -1;
  }

  /**
   * Removes the mapping of the specified value key if this map contains a mapping for the key
   */
  public void remove(int key) {
    int hashKey = getHashKey(key);
    Node head = values[hashKey];
    if (head == null) {
      return;
    }
    if (head.key == key) {
      values[hashKey] = head.next;
      return;
    }
    if (head.next == null) {
      return;
    }
    Node prev = head;
    head = head.next;
    while (head != null) {
      if (head.key == key) {
        prev.next = head.next;
        return;
      }
      prev = head;
      head = head.next;
    }
  }

  class Node {
    Integer key, value;
    Node next;

    Node(Integer key, Integer value) {
      this.key = key;
      this.value = value;
    }
  }

}
