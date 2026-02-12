from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

from .base import TaskBase, labels_from_loss_mask, sample_n_with_inverse_sqrt_bias
from .common import generate_multi_token_words, split_multi_token_words


class _BrevoCore:
    def __init__(
        self,
        n: int,
        *,
        vocab_size: int,
        max_in: int,
        bos_token_id: int,
        eos_token_id: int,
        query_token_id: int,
        answer_token_id: int,
    ) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self.max_in = max_in
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.query_token_id = query_token_id
        self.answer_token_id = answer_token_id

    def generate_tokens(self, rng: random.Random, *, multi: bool) -> tuple[list[int], list[int], list[int], int]:
        dag, topo, depth = self._generate_sample(rng)
        query = topo[-1]

        if multi:
            word_map = self._build_word_map(rng, dag)

        edges = [(p, c) for c, ps in dag.items() for p in ps]
        rng.shuffle(edges)

        tokens = [self.bos_token_id]
        for parent, child in edges:
            if multi:
                tokens += word_map[parent] + word_map[child]
            else:
                tokens += [parent, child]

        if multi:
            tokens += [self.query_token_id] + word_map[query] + [self.answer_token_id]
        else:
            tokens += [self.query_token_id, query, self.answer_token_id]

        loss_mask = [0] * (len(tokens) - 1)
        token_type = [-1, -2 - depth] + [0] * (len(tokens) - 2)
        fake_depth = depth if depth <= 9 else 9

        if multi:
            first = True
            for node in topo:
                tokens += word_map[node]
                if first:
                    token_type += [fake_depth + 1] * len(word_map[node])
                    first = False
                else:
                    token_type += [0] * len(word_map[node])
        else:
            tokens += topo
            token_type += [fake_depth + 1] + [0] * (len(topo) - 1)

        tokens += [self.eos_token_id]
        token_type += [0]
        loss_mask += [1] * (len(tokens) - len(loss_mask))
        return tokens, token_type, loss_mask, depth

    def _generate_sample(self, rng: random.Random):
        nodes, dag = self._generate_dag(rng)
        start_index = max(len(nodes) * 3 // 4, len(nodes) - 1)
        candidate_nodes = nodes[start_index:]
        nonzero_degree_nodes = [node for node in candidate_nodes if len(dag[node]) > 0]
        query = rng.choice(nonzero_degree_nodes)
        subdag = self._subtree_from_query(dag, query)
        topo = self._topological_sort(subdag, rng)
        depth = self._compute_graph_depth(subdag, query)
        return dag, topo, depth

    def _generate_dag(self, rng: random.Random):
        nodes = rng.sample(range(1, self.vocab_size + 1), self.n)
        dag: dict[int, list[int]] = defaultdict(list)
        out_degree = defaultdict(int)
        leaves = rng.randint(1, (len(nodes) - 1) // 4 + 1)

        for i in range(leaves, len(nodes)):
            tgt = nodes[i]
            possible_parents = [src for src in nodes[:i] if out_degree[src] < self.max_in]
            if not possible_parents:
                continue
            num_parents = rng.randint(1, min(len(possible_parents), self.max_in))
            parents = rng.sample(possible_parents, num_parents)
            for parent in parents:
                dag[tgt].append(parent)
                out_degree[parent] += 1
        return nodes, dag

    @staticmethod
    def _subtree_from_query(dag: dict[int, list[int]], query: int) -> dict[int, list[int]]:
        visited = set()
        stack = [query]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for parent in dag.get(node, []):
                if parent not in visited:
                    stack.append(parent)

        filtered: dict[int, list[int]] = defaultdict(list)
        for node in visited:
            for parent in dag.get(node, []):
                if parent in visited:
                    filtered[node].append(parent)
        for node in visited:
            _ = filtered[node]
        return filtered

    @staticmethod
    def _topological_sort(dag: dict[int, list[int]], rng: random.Random) -> list[int]:
        indegree = {node: 0 for node in dag}
        for parents in dag.values():
            for parent in parents:
                indegree[parent] += 1

        queue = [node for node in dag if indegree[node] == 0]
        order: list[int] = []
        while queue:
            node = queue.pop(rng.randint(0, len(queue) - 1))
            order.append(node)
            for parent in dag[node]:
                indegree[parent] -= 1
                if indegree[parent] == 0:
                    queue.append(parent)
        order.reverse()
        return order

    @staticmethod
    def _compute_graph_depth(dag: dict[int, list[int]], query: int) -> int:
        distance = {query: 0}
        queue = deque([query])
        while queue:
            node = queue.popleft()
            for parent in dag[node]:
                if parent not in distance:
                    distance[parent] = distance[node] + 1
                    queue.append(parent)

        leaves = [node for node in dag if len(dag[node]) == 0]
        if not leaves:
            return 0
        return min(distance.get(leaf, float("inf")) for leaf in leaves if leaf in distance)

    @staticmethod
    def _build_word_map(rng: random.Random, dag: dict[int, list[int]]) -> dict[int, list[int]]:
        all_node_ids = sorted(set(dag.keys()) | {p for ps in dag.values() for p in ps})
        id_to_index = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
        word_list = generate_multi_token_words(
            rng,
            n=len(all_node_ids),
            mini_vocab=4,
            min_tlen=2,
            max_tlen=4,
        )
        return {node_id: word_list[id_to_index[node_id]] for node_id in all_node_ids}

    def validate_tokens(self, tokens: Sequence[int], *, multi: bool) -> bool:
        if multi:
            ok, _, _ = self._parse_tokens_multi(tokens)
            return ok
        ok, _, _ = self._parse_tokens(tokens)
        return ok

    def _parse_tokens(self, tokens: Sequence[int]) -> tuple[bool, int | None, list[int] | None]:
        if not tokens or tokens[0] != self.bos_token_id or tokens[-1] != self.eos_token_id:
            return False, None, None

        try:
            idx_query = tokens.index(self.query_token_id)
            idx_answer = tokens.index(self.answer_token_id)
        except ValueError:
            return False, None, None

        edge_tokens = tokens[1:idx_query]
        if len(edge_tokens) % 2 != 0:
            return False, None, None

        edges = [(edge_tokens[i], edge_tokens[i + 1]) for i in range(0, len(edge_tokens), 2)]
        query = tokens[idx_query + 1]
        topo = list(tokens[idx_answer + 1 : -1])

        dag: dict[int, list[int]] = defaultdict(list)
        for parent, child in edges:
            dag[child].append(parent)

        reachable = set()
        stack = [query]
        while stack:
            node = stack.pop()
            reachable.add(node)
            for parent in dag.get(node, []):
                if parent not in reachable:
                    stack.append(parent)

        if set(topo) != reachable:
            return False, query, topo

        seen = set()
        for node in topo:
            for parent in dag.get(node, []):
                if parent not in seen:
                    return False, query, topo
            seen.add(node)

        return True, query, topo

    def _parse_tokens_multi(
        self,
        tokens: Sequence[int],
        mini_vocab: int = 4,
    ) -> tuple[bool, int | None, list[tuple[int, ...]] | None]:
        if not tokens or tokens[0] != self.bos_token_id or tokens[-1] != self.eos_token_id:
            return False, None, None

        try:
            idx_query = tokens.index(self.query_token_id)
            idx_answer = tokens.index(self.answer_token_id)
        except ValueError:
            return False, None, None

        try:
            edge_words = split_multi_token_words(tokens[1:idx_query], mini_vocab)
            query_words = split_multi_token_words(tokens[idx_query + 1 : idx_answer], mini_vocab)
            answer_words = split_multi_token_words(tokens[idx_answer + 1 : -1], mini_vocab)
        except ValueError:
            return False, None, None
        if len(edge_words) % 2 != 0 or len(query_words) != 1:
            return False, None, None

        query_word = query_words[0]
        all_words = set(edge_words + [query_word] + answer_words)
        word_to_id = {w: i for i, w in enumerate(sorted(all_words))}
        id_to_word = {i: w for w, i in word_to_id.items()}

        dag: dict[int, list[int]] = defaultdict(list)
        for i in range(0, len(edge_words), 2):
            parent, child = edge_words[i], edge_words[i + 1]
            dag[word_to_id[child]].append(word_to_id[parent])

        query = word_to_id[query_word]
        topo = [word_to_id[w] for w in answer_words]

        reachable = set()
        stack = [query]
        while stack:
            node = stack.pop()
            reachable.add(node)
            for parent in dag.get(node, []):
                if parent not in reachable:
                    stack.append(parent)

        if set(topo) != reachable:
            return False, query, [id_to_word[i] for i in topo]

        seen = set()
        for node in topo:
            for parent in dag.get(node, []):
                if parent not in seen:
                    return False, query, [id_to_word[i] for i in topo]
            seen.add(node)

        return True, query, [id_to_word[i] for i in topo]


class BrevoTask(TaskBase):
    def _sample_n(self, rng: random.Random) -> int:
        if self.config["qa"]:
            return int(self.config["N"])
        return sample_n_with_inverse_sqrt_bias(rng, int(self.config["N"]))

    def _build_core(self, n: int) -> _BrevoCore:
        cfg = self.config
        return _BrevoCore(
            n=n,
            vocab_size=int(cfg["N"]),
            max_in=int(cfg["max_in"]),
            bos_token_id=int(cfg["bos_token_id"]),
            eos_token_id=int(cfg["eos_token_id"]),
            query_token_id=int(cfg["query_token_id"]),
            answer_token_id=int(cfg["answer_token_id"]),
        )

    def generate_example(self, rng: random.Random) -> dict[str, Any]:
        n = self._sample_n(rng)
        core = self._build_core(n)
        input_ids, token_type, loss_mask, depth = core.generate_tokens(
            rng, multi=bool(self.config["multi"])
        )
        return {
            "input_ids": input_ids,
            "token_type": token_type,
            "loss_mask": loss_mask,
            "labels": labels_from_loss_mask(input_ids, loss_mask),
            "n": n,
            "depth": depth,
            "multi": bool(self.config["multi"]),
        }

    def validate_example(self, example: Mapping[str, Any]) -> bool:
        return self.validate_input_ids(example["input_ids"])

    def validate_input_ids(self, input_ids: Sequence[int]) -> bool:
        core = self._build_core(n=3)
        return core.validate_tokens(input_ids, multi=bool(self.config["multi"]))
