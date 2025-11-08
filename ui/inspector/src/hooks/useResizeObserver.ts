import { MutableRefObject, useLayoutEffect, useRef, useState } from "react";

export interface Size {
  width: number;
  height: number;
}

/**
 * Observe changes to an element's box using ResizeObserver.
 * Keeps React in control of the DOM while avoiding layout thrash.
 */
export function useResizeObserver<T extends HTMLElement>(): [
  MutableRefObject<T | null>,
  Size
] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      const box = entry.contentBoxSize;
      if (Array.isArray(box) && box.length > 0) {
        const { inlineSize, blockSize } = box[0];
        setSize({ width: inlineSize, height: blockSize });
      } else if (
        box &&
        typeof box === "object" &&
        "inlineSize" in box &&
        "blockSize" in box
      ) {
        const { inlineSize, blockSize } = box as {
          inlineSize: number;
          blockSize: number;
        };
        setSize({ width: inlineSize, height: blockSize });
      } else {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}
