import json
from typing import List


class TreeNode:
    def __init__(self, category, children=None, significant=False):
        self.category = category
        self.children = children if children is not None else []
        self.significant = significant

    def get_category(self) -> str:
        return self.category

    def get_children(self) -> List:
        return self.children

    def get_significant(self) -> bool:
        return self.significant

    def add_children(self, child):
        self.children.append(child)


class HierarchyTree:
    def __init__(self, path):
        self.root = TreeNode("Object", significant=True)
        self.load_tree(path)

    def _load_tree(self, data, parent):
        for key in data:
            is_significant = key[-1] != ":"
            category = key.split(":")[0]

            child = TreeNode(category, significant=is_significant)
            parent.add_children(child)

            self._load_tree(data[key], child)

    def load_tree(self, path):
        with open(path, 'r') as f:
            tree_json = json.load(f)

        self._load_tree(tree_json, self.root)

    def find_significant_parent(self, category: str):
        ans = [""]

        def helper(root: TreeNode):
            if root is None:
                return False

            children_found = any(helper(c) for c in root.get_children())
            current_found = root.get_category() == category
            answer_not_set = ans[0] == ""

            if answer_not_set and root.get_significant() and (current_found or children_found):
                ans[0] = root.get_category()

            return children_found or current_found

        helper(self.root)
        return ans[0]

    def find_common_parent(self, categories: List[str]):
        ans = [""]

        def helper(root: TreeNode):
            if root is None:
                return []

            children_found = list(set().union(*[helper(c) for c in root.get_children()]))
            current_found = [root.get_category()] if root.get_category() in categories else []
            all_found = list(set(children_found + current_found))
            answer_not_set = ans[0] == ""

            if answer_not_set and len(all_found) == len(categories):
                ans[0] = root.get_category()

            return all_found

        helper(self.root)
        return ans[0]











