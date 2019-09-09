public class LinkedLists {

  public ListNode detectCycle(ListNode head) {
    if (head == null || head.next == null || head.next.next != null) {
      return null;
    }
    ListNode hare = head, tor = head;
    ListNode intersection = null;
    while (tor.next != null && hare.next.next != null) {
      tor = tor.next;
      hare = hare.next.next;
      if (tor == hare) {
        intersection = tor;
        break;
      }
      if (hare.next == null) {
        break;
      }
    }
    if(intersection == null){
      return null;
    }
    while (head != intersection){
      head = head.next;
      intersection = intersection.next;
    }
    return head;
  }

  public ListNode reverseListRecur(ListNode head) {
    if (head == null || head.next == null) {
      return head;
    }
    ListNode p = reverseListRecur(head.next);
    head.next.next = head;
    head.next = null;
    return p;
  }


  public ListNode reverseList(ListNode head) {
    if (head == null) {
      return head;
    }
    ListNode prev = head, next = head.next;
    prev.next = null;
    while (next != null) {
      ListNode t = next.next;
      next.next = prev;
      prev = next;
      next = t;
    }
    return prev;
  }

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
  }

  ;
}


