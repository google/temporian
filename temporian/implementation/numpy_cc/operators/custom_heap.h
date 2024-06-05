#include <iostream>
#include <list>
template <typename T>
class CustomHeap {
 private:
  std::function<bool(T, T)> compare;
  std::list<T> heap;
  std::unordered_map<T, typename std::list<T>::iterator> val_to_node;

 public:
  CustomHeap(std::function<bool(T, T)> compare) : compare(compare) {}

  void push(T value) {
    heap.push_back(value);
    auto it = std::prev(heap.end());
    // Notice that this breaks if a value repeats, not a problem in our case
    // since we are using the Heap to store the indices
    val_to_node[value] = it;
    // TODO: there is no better way to insert in order with a list
    // but exploring with trees could make this better
    while (it != heap.begin()) {
      auto parent = std::prev(it);
      if (!compare(*parent, *it)) {
        break;
      }
      std::swap(*parent, *it);
      val_to_node[*it] = it;
      val_to_node[*parent] = parent;
      it = parent;
    }
  }

  std::optional<T> pop() {
    if (heap.size() == 0) {
      return {};
    } else {
      auto value = heap.back();
      heap.pop_back();
      auto it = val_to_node.find(value);
      // all other pointers in val_to_node are still valid because
      // heap is a double linked list
      val_to_node.erase(it);
      return value;
    }
  }

  std::optional<T> top() {
    if (heap.empty()) {
      return {};
    } else {
      return heap.back();
    }
  }

  void remove(T value) {
    auto it = val_to_node.find(value);
    if (it != val_to_node.end()) {
      heap.erase(it->second);
      // all other pointers in val_to_node are still valid because
      // heap is a double linked list
      val_to_node.erase(it);
    }
  }
  int size() { return heap.size(); }
  int empty() { return heap.empty(); }
};
