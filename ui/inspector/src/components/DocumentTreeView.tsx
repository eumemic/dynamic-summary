import { useEffect, useMemo, useRef, useState } from "react";
import { useDocumentNodes } from "../hooks/useDocumentNodes";
import TreeCanvas from "./TreeCanvas";
import NodeDetailsPanel from "./NodeDetailsPanel";

interface DocumentTreeViewProps {
  documentId: string;
}

const DEFAULT_WINDOW = 2000;
const clampLimit = (value: number) => Math.max(1, Math.min(2000, value));
const numberFormatter = new Intl.NumberFormat();
const MIN_WINDOW = 50;

function normalizeSpan(start: number, end: number): { start: number; end: number } {
  if (end <= start) {
    return { start, end: start + 1 };
  }
  return { start, end };
}

export default function DocumentTreeView({
  documentId,
}: DocumentTreeViewProps) {
  const [spanStart, setSpanStart] = useState(0);
  const [spanEnd, setSpanEnd] = useState(DEFAULT_WINDOW);
  const [limit, setLimit] = useState(200);
  const [minHeight, setMinHeight] = useState<number | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hasManualRange, setHasManualRange] = useState(false);
  const panAnchorRef = useRef<number | null>(null);
  const panRangeRef = useRef<number>(spanEnd - spanStart);

  useEffect(() => {
    setSpanStart(0);
    setSpanEnd(DEFAULT_WINDOW);
    setSelectedNodeId(null);
    setHoveredNodeId(null);
    setHasManualRange(false);
    panAnchorRef.current = null;
  }, [documentId]);

  const span = useMemo(
    () => normalizeSpan(spanStart, spanEnd),
    [spanStart, spanEnd]
  );

  const { nodes, totalMatching, loading, error, refresh } = useDocumentNodes({
    documentId,
    spanStart: span.start,
    spanEnd: span.end,
    limit,
    minHeight,
  });

  const documentSpanEnd = useMemo(
    () =>
      nodes.reduce<number>(
        (acc, node) => (node.span_end > acc ? node.span_end : acc),
        0
      ),
    [nodes]
  );

  useEffect(() => {
    if (!hasManualRange && documentSpanEnd > 0) {
      setSpanStart(0);
      setSpanEnd(documentSpanEnd);
    }
  }, [documentSpanEnd, hasManualRange]);

  useEffect(() => {
    if (documentSpanEnd <= 0) {
      return;
    }
    setSpanEnd((current) =>
      current > documentSpanEnd ? documentSpanEnd : current
    );
    setSpanStart((current) =>
      current >= documentSpanEnd
        ? Math.max(0, documentSpanEnd - 1)
        : current
    );
  }, [documentSpanEnd]);

  useEffect(() => {
    if (panAnchorRef.current !== null) {
      panRangeRef.current = span.end - span.start;
    }
  }, [span.start, span.end]);

  useEffect(() => {
    if (
      selectedNodeId &&
      !nodes.some((node) => node.node_id === selectedNodeId)
    ) {
      setSelectedNodeId(null);
    }
    if (
      hoveredNodeId &&
      !nodes.some((node) => node.node_id === hoveredNodeId)
    ) {
      setHoveredNodeId(null);
    }
  }, [nodes, selectedNodeId, hoveredNodeId]);

  const selectedNode = useMemo(
    () => nodes.find((node) => node.node_id === selectedNodeId) ?? null,
    [nodes, selectedNodeId]
  );

  const hoveredNode = useMemo(
    () => nodes.find((node) => node.node_id === hoveredNodeId) ?? null,
    [nodes, hoveredNodeId]
  );

  const sliderMax =
    documentSpanEnd > 0 ? documentSpanEnd : Math.max(spanEnd, spanStart + 1);

  const applySpanUpdate = (nextStart: number, nextEnd: number) => {
    setHasManualRange(true);

    let clampedStart = Math.max(0, nextStart);
    let clampedEnd = Math.max(nextStart + 1, nextEnd);

    if (sliderMax > 0) {
      clampedEnd = Math.min(clampedEnd, sliderMax);
      clampedStart = Math.max(0, Math.min(clampedStart, clampedEnd - 1));
    }

    const roundedStart = Math.floor(clampedStart);
    const roundedEnd = Math.max(roundedStart + 1, Math.ceil(clampedEnd));

    setSpanStart(roundedStart);
    setSpanEnd(roundedEnd);
  };

  const handleSpanStartChange = (value: number) => {
    setHasManualRange(true);
    const next = Math.min(value, spanEnd - 1);
    setSpanStart(Math.max(0, next));
  };

  const handleSpanEndChange = (value: number) => {
    setHasManualRange(true);
    const next = Math.max(value, spanStart + 1);
    setSpanEnd(Math.min(next, sliderMax));
  };

  const handleLimitChange = (value: number) => {
    setLimit(clampLimit(value));
  };

  const handleMinHeightChange = (raw: string) => {
    if (raw === "") {
      setMinHeight(null);
      return;
    }
    const parsed = Math.max(0, Number(raw));
    setMinHeight(Number.isNaN(parsed) ? null : parsed);
  };

  const handleRefresh = () => {
    setHasManualRange(true);
    refresh();
  };

  const handleZoom = (centerRatio: number, deltaY: number) => {
    if (!Number.isFinite(centerRatio)) {
      return;
    }

    const currentRange = span.end - span.start;
    if (currentRange <= 0) {
      return;
    }

    const ratio = Math.min(Math.max(centerRatio, 0), 1);
    const zoomFactor = Math.exp(deltaY * 0.0015);
    const rawRange = currentRange * zoomFactor;
    const maxRange = sliderMax > 0 ? sliderMax : Math.max(currentRange, DEFAULT_WINDOW);
    const nextRange = Math.min(
      Math.max(rawRange, MIN_WINDOW),
      Math.max(maxRange, MIN_WINDOW)
    );

    const target = span.start + ratio * currentRange;
    let nextStart = target - ratio * nextRange;
    let nextEnd = nextStart + nextRange;

    if (sliderMax > 0) {
      if (nextStart < 0) {
        nextStart = 0;
        nextEnd = nextRange;
      }
      if (nextEnd > sliderMax) {
        nextEnd = sliderMax;
        nextStart = Math.max(0, nextEnd - nextRange);
      }
    }

    if (nextEnd - nextStart < MIN_WINDOW) {
      nextEnd = nextStart + MIN_WINDOW;
    }

    applySpanUpdate(nextStart, nextEnd);
  };

  const handlePanStart = (ratio: number) => {
    const range = span.end - span.start;
    if (range <= 0) {
      panAnchorRef.current = null;
      return;
    }
    const clamped = Math.min(Math.max(ratio, 0), 1);
    panAnchorRef.current = span.start + clamped * range;
    panRangeRef.current = range;
    setHasManualRange(true);
  };

  const handlePanMove = (ratio: number) => {
    const anchor = panAnchorRef.current;
    if (anchor === null) {
      return;
    }
    const range = Math.max(panRangeRef.current, 1);
    const clamped = Math.min(Math.max(ratio, 0), 1);
    let nextStart = anchor - clamped * range;
    let nextEnd = nextStart + range;

    if (sliderMax > 0) {
      if (nextStart < 0) {
        nextStart = 0;
        nextEnd = range;
      }
      if (nextEnd > sliderMax) {
        nextEnd = sliderMax;
        nextStart = Math.max(0, nextEnd - range);
      }
    }

    applySpanUpdate(nextStart, nextEnd);
  };

  const handlePanEnd = () => {
    panAnchorRef.current = null;
  };

  const formatNumber = (value: number) =>
    numberFormatter.format(Math.round(value));

  const renderStatus = () => {
    if (loading) {
      return <span>Loading nodes…</span>;
    }
    if (error) {
      return <span style={{ color: "#ff6b6b" }}>{error}</span>;
    }
    return (
      <span>
        Showing {nodes.length} of {totalMatching} nodes covering span [
        {formatNumber(span.start)}, {formatNumber(span.end)}).
      </span>
    );
  };

  const handleHover = (nodeId: string | null) => {
    setHoveredNodeId(nodeId);
  };

  const handleSelect = (nodeId: string) => {
    setSelectedNodeId(nodeId);
  };

  return (
    <section className="document-view">
      <header className="controls">
        <div className="span-controls">
          <div className="span-controls__header">
            <div>
              <strong>Span (character offsets)</strong>
              <div className="span-controls__range">
                [{formatNumber(span.start)}, {formatNumber(span.end)}) •{" "}
                {formatNumber(span.end - span.start)} chars
              </div>
            </div>
            <div className="span-controls__inputs">
              <label>
                Start
                <input
                  type="number"
                  min={0}
                  max={spanEnd - 1}
                  value={spanStart}
                  onChange={(event) =>
                    handleSpanStartChange(Number(event.target.value))
                  }
                />
              </label>
              <label>
                End
                <input
                  type="number"
                  min={spanStart + 1}
                  max={sliderMax}
                  value={spanEnd}
                  onChange={(event) =>
                    handleSpanEndChange(Number(event.target.value))
                  }
                />
              </label>
            </div>
          </div>
          <div className="span-slider">
            <input
              type="range"
              min={0}
              max={sliderMax}
              value={spanStart}
              onChange={(event) =>
                handleSpanStartChange(Number(event.target.value))
              }
            />
            <input
              type="range"
              min={0}
              max={sliderMax}
              value={spanEnd}
              onChange={(event) =>
                handleSpanEndChange(Number(event.target.value))
              }
            />
          </div>
        </div>
        <div className="render-controls">
          <label>
            Node budget
            <input
              type="number"
              min={1}
              max={2000}
              value={limit}
              onChange={(event) => handleLimitChange(Number(event.target.value))}
            />
          </label>
          <label>
            Min height
            <input
              type="number"
              min={0}
              placeholder="Any"
              value={minHeight ?? ""}
              onChange={(event) => handleMinHeightChange(event.target.value)}
            />
          </label>
          <button type="button" onClick={handleRefresh}>
            Refresh
          </button>
        </div>
      </header>

      <div className="status">{renderStatus()}</div>

      <div className="tree-layout">
        <TreeCanvas
          nodes={nodes}
          spanStart={span.start}
          spanEnd={span.end}
          maxSpanEnd={documentSpanEnd}
          selectedNodeId={selectedNodeId}
          hoveredNodeId={hoveredNodeId}
          onHover={handleHover}
          onSelect={handleSelect}
          onZoom={handleZoom}
          onPanStart={handlePanStart}
          onPanMove={handlePanMove}
          onPanEnd={handlePanEnd}
        />
        <NodeDetailsPanel node={selectedNode} hoveredNode={hoveredNode} />
      </div>
    </section>
  );
}
