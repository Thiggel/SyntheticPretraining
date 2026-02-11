# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu (original release)
# Refactor: self-contained HF dataset generator + validation utilities

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hf_dataset_utils import (
    labels_from_loss_mask,
    make_hf_dataset,
    sample_n_with_inverse_sqrt_bias,
)


def generate_multi_token_words(
    rng: random.Random,
    n: int,
    mini_vocab: int = 3,
    min_tlen: int = 5,
    max_tlen: int = 7,
) -> List[List[int]]:
    def sample_word(length: int) -> Tuple[int, ...]:
        toks = [rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab
        return tuple(toks)

    words = set()
    while len(words) < n:
        words.add(sample_word(rng.randint(min_tlen, max_tlen)))
    return [list(word) for word in words]


class TopoSortDepthStats:
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

    def generate_dag(self, rng: random.Random):
        nodes = rng.sample(range(1, self.vocab_size + 1), self.n)
        dag: Dict[int, List[int]] = defaultdict(list)
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
    def subtree_from_query(dag: Dict[int, List[int]], query: int) -> Dict[int, List[int]]:
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

        filtered: Dict[int, List[int]] = defaultdict(list)
        for node in visited:
            for parent in dag.get(node, []):
                if parent in visited:
                    filtered[node].append(parent)
        for node in visited:
            _ = filtered[node]
        return filtered

    @staticmethod
    def topological_sort(dag: Dict[int, List[int]], rng: random.Random) -> List[int]:
        indegree = {node: 0 for node in dag}
        for node, parents in dag.items():
            for parent in parents:
                indegree[parent] += 1

        queue = [node for node in dag if indegree[node] == 0]
        order: List[int] = []
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
    def compute_graph_depth(dag: Dict[int, List[int]], query: int) -> int:
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

    def generate_sample(self, rng: random.Random):
        nodes, dag = self.generate_dag(rng)
        start_index = max(len(nodes) * 3 // 4, len(nodes) - 1)
        candidate_nodes = nodes[start_index:]
        nonzero_degree_nodes = [node for node in candidate_nodes if len(dag[node]) > 0]
        query = rng.choice(nonzero_degree_nodes)

        subdag = self.subtree_from_query(dag, query)
        topo = self.topological_sort(subdag, rng)
        depth = self.compute_graph_depth(subdag, query)
        return dag, topo, depth

    def generate_tokens(self, rng: random.Random, multi: bool = False):
        dag, topo, depth = self.generate_sample(rng)
        query = topo[-1]

        if multi:
            all_node_ids = sorted(set(dag.keys()) | {p for ps in dag.values() for p in ps})
            id_to_index = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            word_list = generate_multi_token_words(
                rng,
                n=len(all_node_ids),
                mini_vocab=4,
                min_tlen=2,
                max_tlen=4,
            )
            word_map = {node_id: word_list[id_to_index[node_id]] for node_id in all_node_ids}

        edges = [(p, c) for c, ps in dag.items() for p in ps]
        rng.shuffle(edges)

        tokens = [self.bos_token_id]
        for p, c in edges:
            if multi:
                tokens += word_map[p] + word_map[c]
            else:
                tokens += [p, c]

        if multi:
            tokens += [self.query_token_id] + word_map[query] + [self.answer_token_id]
        else:
            tokens += [self.query_token_id, query, self.answer_token_id]

        list_label = [0] * (len(tokens) - 1)
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
        list_label += [1] * (len(tokens) - len(list_label))
        return tokens, token_type, list_label, depth

    def parse_tokens(self, tokens: Sequence[int]) -> Tuple[bool, int | None, List[int] | None]:
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

        dag: Dict[int, List[int]] = defaultdict(list)
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

    def parse_tokens_multi(
        self,
        tokens: Sequence[int],
        mini_vocab: int = 4,
    ) -> Tuple[bool, int | None, List[Tuple[int, ...]] | None]:
        if not tokens or tokens[0] != self.bos_token_id or tokens[-1] != self.eos_token_id:
            return False, None, None

        try:
            idx_query = tokens.index(self.query_token_id)
            idx_answer = tokens.index(self.answer_token_id)
        except ValueError:
            return False, None, None

        def split_words(token_seq: Sequence[int]) -> List[Tuple[int, ...]]:
            words: List[Tuple[int, ...]] = []
            word: List[int] = []
            for tok in token_seq:
                word.append(tok)
                if mini_vocab < tok <= 2 * mini_vocab:
                    words.append(tuple(word))
                    word = []
            if word:
                return []
            return words

        edge_words = split_words(tokens[1:idx_query])
        query_words = split_words(tokens[idx_query + 1 : idx_answer])
        answer_words = split_words(tokens[idx_answer + 1 : -1])
        if len(edge_words) % 2 != 0 or len(query_words) != 1:
            return False, None, None

        query_word = query_words[0]
        all_words = set(edge_words + [query_word] + answer_words)
        word_to_id = {w: i for i, w in enumerate(sorted(all_words))}
        id_to_word = {i: w for w, i in word_to_id.items()}

        dag: Dict[int, List[int]] = defaultdict(list)
        for i in range(0, len(edge_words), 2):
            p, c = edge_words[i], edge_words[i + 1]
            dag[word_to_id[c]].append(word_to_id[p])

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


def generate_brevo_example(rng: random.Random, config: Mapping[str, Any]) -> Dict[str, object]:
    cfg = dict(config)
    n = cfg["N"] if cfg["qa"] else sample_n_with_inverse_sqrt_bias(rng, cfg["N"])
    topo = TopoSortDepthStats(
        n,
        vocab_size=cfg["N"],
        max_in=cfg["max_in"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        query_token_id=cfg["query_token_id"],
        answer_token_id=cfg["answer_token_id"],
    )

    input_ids, token_type, loss_mask, depth = topo.generate_tokens(rng, multi=cfg["multi"])
    return {
        "input_ids": input_ids,
        "token_type": token_type,
        "loss_mask": loss_mask,
        "labels": labels_from_loss_mask(input_ids, loss_mask),
        "n": n,
        "depth": depth,
        "multi": cfg["multi"],
    }


def validate_brevo_example(input_ids: Sequence[int], config: Mapping[str, Any]) -> bool:
    cfg = dict(config)
    parser = TopoSortDepthStats(
        n=3,
        vocab_size=cfg["N"],
        max_in=cfg["max_in"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        query_token_id=cfg["query_token_id"],
        answer_token_id=cfg["answer_token_id"],
    )
    if cfg["multi"]:
        ok, _, _ = parser.parse_tokens_multi(input_ids)
    else:
        ok, _, _ = parser.parse_tokens(input_ids)
    return ok


def make_brevo_hf_dataset(num_examples: int, config: Mapping[str, Any], seed: int = 42):
    cfg = dict(config)
    return make_hf_dataset(
        num_examples=num_examples,
        seed=seed,
        sample_fn=lambda rng: generate_brevo_example(rng, cfg),
    )
