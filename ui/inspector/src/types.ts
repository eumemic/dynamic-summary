export interface DocumentInfo {
  document_id: string;
  file_path: string | null;
  indexed_at: string;
  node_count: number;
}

export interface DocumentsResponse {
  documents: DocumentInfo[];
}

export interface DocumentsStreamEvent {
  event: "documents_changed";
  documents: DocumentInfo[];
  added_ids: string[];
  removed_ids: string[];
}

export interface NodeResponse {
  node_id: string;
  document_id: string | null;
  parent_id: string | null;
  left_child_id: string | null;
  right_child_id: string | null;
  span_start: number;
  span_end: number;
  text: string;
  token_count: number;
  height: number;
  level_index: number;
  preceding_neighbor_id: string | null;
  following_neighbor_id: string | null;
  is_pinned: boolean;
  created_at: string | null;
}

export interface NodesPageResponse {
  nodes: NodeResponse[];
  total_matching: number;
}

export interface QueryListItem {
  id: string;
  document_id: string;
  query_text: string;
  budget_tokens: number | null;
  num_seeds: number | null;
  created_at: string;
}

export interface DocumentQueriesResponse {
  queries: QueryListItem[];
}

export interface QueryNodeEntry {
  node_id: string;
  score: number;
  is_seed: boolean;
  position: number;
}

export interface QueryDetailResponse {
  query: QueryListItem;
  nodes: QueryNodeEntry[];
}

export interface ExecuteQueryResponse {
  summary: string;
  token_count: number;
  nodes_retrieved: number;
  tiling_size: number;
  query_id: string;
}

export type TelemetryEvent =
  | {
      event: "node_committed";
      node_id: string;
      height: number;
      span_start: number;
      span_end: number;
    }
  | {
      event: "nodes_deleted";
      node_ids: string[];
    }
  | {
      event: "append_started" | "append_completed" | "append_failed";
    };
