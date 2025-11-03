import { useCallback, useMemo, useState } from "react";
import { NodeResponse } from "../types";
import { useResizeObserver } from "../hooks/useResizeObserver";

interface TreeCanvasProps {
  nodes: NodeResponse[];
  spanStart: number;
  spanEnd: number;
  maxSpanEnd: number;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  onHover: (nodeId: string | null) => void;
  onSelect: (nodeId: string) => void;
  onZoom?: (centerRatio: number, deltaY: number) => void;
  onPanStart?: (anchorRatio: number) => void;
  onPanMove?: (currentRatio: number) => void;
  onPanEnd?: () => void;
}

const PADDING = {
  top: 36,
  right: 32,
  bottom: 48,
  left: 48,
};

const MIN_NODE_PIXEL_WIDTH = 4;
const MIN_ROW_HEIGHT = 24;
const MAX_ROW_HEIGHT = 48;
const LABEL_HORIZONTAL_PADDING = 12;
const LABEL_VERTICAL_PADDING = 6;

const formatter = new Intl.NumberFormat();
const COLORS = [
  "#3da9fc",
  "#ef4565",
  "#94a1b2",
  "#2cb67d",
  "#f25f4c",
  "#7f5af0",
  "#ff8906",
];

interface RenderNode {
  node: NodeResponse;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

export default function TreeCanvas({
  nodes,
  spanStart,
  spanEnd,
  maxSpanEnd,
  selectedNodeId,
  hoveredNodeId,
  onHover,
  onSelect,
  onZoom,
  onPanStart,
  onPanMove,
  onPanEnd,
}: TreeCanvasProps) {
  const [containerRef, size] = useResizeObserver<HTMLDivElement>();
  const maxHeight = useMemo(
    () => nodes.reduce((acc, node) => Math.max(acc, node.height), 0),
    [nodes]
  );

  const canvasWidth = Math.max(size.width, 1);
  const canvasHeight = Math.max(size.height, 1);
  const innerWidth = Math.max(canvasWidth - PADDING.left - PADDING.right, 1);
  const innerHeight = Math.max(canvasHeight - PADDING.top - PADDING.bottom, 1);
  const domainSpan = Math.max(spanEnd - spanStart, 1);
  const levelStride = maxHeight === 0 ? innerHeight : innerHeight / (maxHeight + 1);
  const rowHeight = Math.min(
    Math.max(levelStride * 0.6, MIN_ROW_HEIGHT),
    MAX_ROW_HEIGHT
  );

  const renderNodes: RenderNode[] = useMemo(() => {
    if (nodes.length === 0) {
      return [];
    }
    return nodes.map((node) => {
      const clampedStart = Math.max(node.span_start, spanStart);
      const clampedEnd = Math.min(node.span_end, spanEnd);
      const visibleSpan = Math.max(clampedEnd - clampedStart, 0);

      const rawWidth = (visibleSpan / domainSpan) * innerWidth;
      const width =
        visibleSpan === 0
          ? MIN_NODE_PIXEL_WIDTH
          : Math.max(rawWidth, MIN_NODE_PIXEL_WIDTH);
      const offsetStart = Math.max(clampedStart - spanStart, 0);
      const x =
        PADDING.left +
        (domainSpan === 0 ? 0 : (offsetStart / domainSpan) * innerWidth);

      const level =
        maxHeight === 0
          ? 0
          : Math.max(Math.min(node.height, maxHeight), 0);
      const bandCenter =
        PADDING.top + innerHeight - levelStride * level - levelStride / 2;
      const y = bandCenter - rowHeight / 2;

      const collapsedText = node.text.replace(/\s+/g, " ").trim();
      const label = collapsedText === "" ? "(empty node)" : collapsedText;

      return {
        node,
        x,
        y,
        width,
        height: rowHeight,
        label,
      };
    });
  }, [
    nodes,
    spanStart,
    spanEnd,
    domainSpan,
    innerWidth,
    innerHeight,
    rowHeight,
    maxHeight,
    levelStride,
  ]);

  const ticks = useMemo(() => {
    const values: number[] = [];
    const tickCount = 5;
    for (let i = 0; i <= tickCount; i += 1) {
      const ratio = i / tickCount;
      values.push(Math.round(spanStart + ratio * (spanEnd - spanStart)));
    }
    return values;
  }, [spanStart, spanEnd]);

  const docExtent =
    maxSpanEnd === 0
      ? `${formatter.format(spanStart)} – ${formatter.format(spanEnd)}`
      : `${formatter.format(spanStart)} – ${formatter.format(maxSpanEnd)}`;

  const [isPanning, setIsPanning] = useState(false);
  const [activePointerId, setActivePointerId] = useState<number | null>(null);

  const handleWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      if (!onZoom || innerWidth <= 0) {
        return;
      }

      const rect = event.currentTarget.getBoundingClientRect();
      const relativeX = event.clientX - rect.left - PADDING.left;
      const ratio = Math.min(Math.max(relativeX / innerWidth, 0), 1);
      const { deltaMode, deltaY } = event;
      let deltaPixels = deltaY;
      if (deltaMode === WheelEvent.DOM_DELTA_LINE) {
        deltaPixels *= 40;
      } else if (deltaMode === WheelEvent.DOM_DELTA_PAGE) {
        deltaPixels *= innerHeight;
      }

      onZoom(ratio, deltaPixels);
      if (event.cancelable) {
        event.preventDefault();
      }
      event.stopPropagation();
    },
    [onZoom, innerWidth, innerHeight]
  );

  const computeRatio = useCallback(
    (clientX: number, rect: DOMRect) => {
      if (innerWidth <= 0) {
        return 0.5;
      }
      const localX = clientX - rect.left - PADDING.left;
      return Math.min(Math.max(localX / innerWidth, 0), 1);
    },
    [innerWidth]
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0 || innerWidth <= 0) {
        return;
      }
      const rect = event.currentTarget.getBoundingClientRect();
      const ratio = computeRatio(event.clientX, rect);
      event.currentTarget.setPointerCapture(event.pointerId);
      setActivePointerId(event.pointerId);
      setIsPanning(true);
      onPanStart?.(ratio);
    },
    [computeRatio, onPanStart, innerWidth]
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!isPanning || activePointerId !== event.pointerId) {
        return;
      }
      const rect = event.currentTarget.getBoundingClientRect();
      const ratio = computeRatio(event.clientX, rect);
      onPanMove?.(ratio);
      if (event.cancelable) {
        event.preventDefault();
      }
    },
    [isPanning, activePointerId, computeRatio, onPanMove]
  );

  const handlePointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (isPanning && activePointerId === event.pointerId) {
        setIsPanning(false);
        setActivePointerId(null);
        onPanEnd?.();
      }
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [isPanning, activePointerId, onPanEnd]
  );

  const handlePointerLeave = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (isPanning && activePointerId === event.pointerId) {
        setIsPanning(false);
        setActivePointerId(null);
        onPanEnd?.();
      }
    },
    [isPanning, activePointerId, onPanEnd]
  );

  const handlePointerCancel = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (isPanning && activePointerId === event.pointerId) {
        setIsPanning(false);
        setActivePointerId(null);
        onPanEnd?.();
      }
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [isPanning, activePointerId, onPanEnd]
  );

  return (
    <div
      className="tree-canvas"
      ref={containerRef}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onPointerCancel={handlePointerCancel}
    >
      <svg
        width="100%"
        height="100%"
        role="list"
        aria-label={`Tree view spanning ${docExtent}`}
      >
        <defs>
          <linearGradient id="tree-bg" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.05)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.02)" />
          </linearGradient>
        </defs>
        <rect
          x={0}
          y={0}
          width={canvasWidth}
          height={canvasHeight}
          fill="url(#tree-bg)"
          rx={12}
        />

        {Array.from({ length: maxHeight + 1 }).map((_, index) => {
          const bandCenter =
            PADDING.top + innerHeight - levelStride * index - levelStride / 2;
          const y = bandCenter - rowHeight / 2;
          return (
            <g key={index}>
              <line
                x1={PADDING.left}
                x2={canvasWidth - PADDING.right}
                y1={bandCenter}
                y2={bandCenter}
                stroke="rgba(255,255,255,0.05)"
              />
              <text
                x={PADDING.left - 12}
                y={bandCenter}
                textAnchor="end"
                fontSize={12}
                fill="rgba(255,255,255,0.4)"
                dominantBaseline="middle"
                alignmentBaseline="middle"
              >
                h={index}
              </text>
            </g>
          );
        })}

        {renderNodes.map(({ node, x, y, width, height, label }) => {
          const isSelected = node.node_id === selectedNodeId;
          const isHovered = node.node_id === hoveredNodeId;
          const color = node.is_pinned
            ? "#f9c74f"
            : COLORS[node.height % COLORS.length];
          const opacity = isHovered || isSelected ? 1 : 0.75;

          const showLabel = width > 72 && height > MIN_ROW_HEIGHT - 2;
          const textBoxWidth = Math.max(1, width - LABEL_HORIZONTAL_PADDING);
          const textContent = showLabel ? label : "";

          return (
            <g
              key={node.node_id}
              transform={`translate(${x}, ${y})`}
              role="listitem"
              aria-label={`Node ${node.node_id} spanning ${node.span_start} to ${node.span_end}`}
            >
              <rect
                width={width}
                height={height}
                rx={6}
                ry={6}
                fill={color}
                opacity={opacity}
                stroke={isSelected ? "#fffffe" : isHovered ? "#eef4ff" : "none"}
                strokeWidth={isSelected ? 2 : 1}
                tabIndex={0}
                onMouseEnter={() => onHover(node.node_id)}
                onMouseLeave={() => onHover(null)}
                onFocus={() => onHover(node.node_id)}
                onBlur={() => onHover(null)}
                onClick={() => onSelect(node.node_id)}
              >
                <title>{`${label}\nHeight: ${node.height}\nSpan: [${node.span_start}, ${node.span_end})\nTokens: ${node.token_count}`}</title>
              </rect>
              {showLabel && (
                <foreignObject
                  x={LABEL_HORIZONTAL_PADDING / 2}
                  y={LABEL_VERTICAL_PADDING / 2}
                  width={textBoxWidth}
                  height={Math.max(10, height - LABEL_VERTICAL_PADDING)}
                  pointerEvents="none"
                  xmlns="http://www.w3.org/1999/xhtml"
                >
                  <div className="tree-node__label">
                    {textContent}
                  </div>
                </foreignObject>
              )}
            </g>
          );
        })}

        <line
          x1={PADDING.left}
          x2={canvasWidth - PADDING.right}
          y1={canvasHeight - PADDING.bottom}
          y2={canvasHeight - PADDING.bottom}
          stroke="rgba(255,255,255,0.35)"
          strokeWidth={1}
        />

        {ticks.map((tick) => {
          const x =
            PADDING.left +
            ((tick - spanStart) / (spanEnd - spanStart || 1)) * innerWidth;
          return (
            <g key={tick}>
              <line
                x1={x}
                x2={x}
                y1={canvasHeight - PADDING.bottom}
                y2={canvasHeight - PADDING.bottom + 6}
                stroke="rgba(255,255,255,0.35)"
              />
              <text
                x={x}
                y={canvasHeight - PADDING.bottom + 20}
                textAnchor="middle"
                fontSize={12}
                fill="rgba(255,255,255,0.6)"
              >
                {formatter.format(tick)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
