public class LinkedLists {

  public Node insert(Node head, int insertVal) {
    Node root = head;
    if (root == null) {
      root = new Node(insertVal, null);
      root.next = root;
      return root;
    }
    while (true) {
      if (head.val > head.next.val) {
        if (insertVal >= head.val || insertVal <= head.next.val) {
          head.next = new Node(insertVal, head.next);
          break;
        }
      } else {
        if (insertVal >= head.val && insertVal <= head.next.val) {
          head.next = new Node(insertVal, head.next);
          break;
        }
      }
      head = head.next;
      if (head == root) {
        // will reach here if all nodes are equal
        head.next = new Node(insertVal, head.next);
        break;
      }
    }
    return root;
  }

  public ListNode oddEvenList(ListNode root) {
    if (root == null) {
      return root;
    }
    ListNode oddTail = root, evenHead = root.next, evenTail = root.next;

    while (oddTail != null && evenTail != null && evenTail.next != null) {
      oddTail.next = evenTail.next;
      oddTail = oddTail.next;
      evenTail.next = oddTail.next;
      evenTail = evenTail.next;
    }
    oddTail.next = evenHead;
    return root;
  }

  class Node {
    public int val;
    public Node next;

    public Node() {
    }

    public Node(int _val, Node _next) {
      val = _val;
      next = _next;
    }
  };
}

