import { DocumentsResponse, NodeResponse, NodesPageResponse } from "../types";

const API_BASE =
  import.meta.env.VITE_RAGZOOM_API_URL?.replace(/\/$/, "") ?? "";

const JSON_HEADERS = {
  Accept: "application/json",
};

async function requestJson<T>(
  input: RequestInfo,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(input, {
    ...init,
    headers: {
      ...JSON_HEADERS,
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      detail || `Request failed with status ${response.status}`
    );
  }

  return (await response.json()) as T;
}

export async function fetchDocuments(): Promise<DocumentsResponse> {
  return requestJson<DocumentsResponse>(`${API_BASE}/documents`);
}

export async function fetchNodesInSpan(
  documentId: string,
  spanStart: number,
  spanEnd: number,
  limit: number,
  minHeight?: number | null
): Promise<NodesPageResponse> {
  const params = new URLSearchParams({
    span_start: spanStart.toString(),
    span_end: spanEnd.toString(),
    limit: limit.toString(),
  });
  if (minHeight !== undefined && minHeight !== null) {
    params.set("min_height", minHeight.toString());
  }
  return requestJson<NodesPageResponse>(
    `${API_BASE}/documents/${encodeURIComponent(
      documentId
    )}/nodes?${params.toString()}`
  );
}

export async function fetchNodesBatch(
  documentId: string,
  nodeIds: string[]
): Promise<NodeResponse[]> {
  if (nodeIds.length === 0) {
    return [];
  }

  const payload = { node_ids: nodeIds };
  const response = await requestJson<{ nodes: NodeResponse[] }>(
    `${API_BASE}/documents/${encodeURIComponent(
      documentId
    )}/nodes/batch`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    }
  );

  return response.nodes;
}
