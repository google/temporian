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
    val_to_node[value] = it;
    // TODO: better sorting?
    while (it != heap.begin()) {
      auto parent = std::prev(it);
      if (!compare(*parent, *it)) {
        break;
      }
      // TODO: check that this swap is doing what I want
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
      val_to_node.erase(it);
    } else {
      // TODO: exception meant for debugging, remove it
      throw std::invalid_argument("removing a value that doesn't exists");
    }
  }
  int size() { return heap.size(); }
  int empty() { return heap.empty(); }

  void print() {
    std::cout << "my_heap{" << std::endl << " [";
    std::for_each(heap.begin(), heap.end(),
                  [](const int n) { std::cout << n << ' '; });
    std::cout << "]" << std::endl;

    // std::cout << " {" << std::endl;
    // for (const auto& pair : val_to_node) {
    //   std::cout << "  " << pair.first << ": " << *(pair.second) << std::endl;
    // }
    // std::cout << " }" << std::endl;
  }
};
