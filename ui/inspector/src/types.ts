export interface DocumentInfo {
  document_id: string;
  file_path: string | null;
  indexed_at: string;
  node_count: number;
}

export interface DocumentsResponse {
  documents: DocumentInfo[];
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
