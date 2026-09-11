import type { ReactNode } from "react";

/**
 * 左右两侧的示意性聚合物主链：之字骨架 + 节点，配合 CSS 弹跳/摆动。
 * 非化学精确结构，仅作氛围装饰。
 */
function buildChainElements(
  n: number,
  opts: { cxBase: number; step: number; mirrorX: boolean },
) {
  const { cxBase, step, mirrorX } = opts;
  const lines: ReactNode[] = [];
  const nodes: ReactNode[] = [];
  let prev = { x: cxBase, y: 6 };

  for (let i = 0; i < n; i++) {
    const y = 6 + i * step;
    const zig = Math.sin(i * 0.42) * 11 + Math.sin(i * 0.19) * 4;
    const x = cxBase + (mirrorX ? -zig : zig);

    if (i > 0) {
      lines.push(
        <line
          key={`b-${i}`}
          x1={prev.x}
          y1={prev.y}
          x2={x}
          y2={y}
          stroke="#c5c5ca"
          strokeWidth={1.2}
          strokeLinecap="round"
        />,
      );
    }
    prev = { x, y };

    const isHetero = i % 9 === 0;
    const isBranch = i % 4 === 2;
    const r = isHetero ? 4.8 : isBranch ? 3.4 : 3.1;
    const fill = isHetero ? "#0071e3" : isBranch ? "#b8b8be" : "#d8d8de";
    const stroke = isHetero ? "#005bb5" : "#a8a8ae";

    nodes.push(
      <circle
        key={`a-${i}`}
        cx={x}
        cy={y}
        r={r}
        fill={fill}
        fillOpacity={isHetero ? 0.35 : 0.75}
        stroke={stroke}
        strokeWidth={0.55}
        className="polymer-atom"
        style={{ animationDelay: `${(i * 0.085) % 2}s` }}
      />,
    );

    if (isBranch && i < n - 2) {
      const bx = x + (mirrorX ? -14 : 14);
      const by = y + step * 0.35;
      lines.push(
        <line
          key={`br-${i}`}
          x1={x}
          y1={y}
          x2={bx}
          y2={by}
          stroke="#d2d2d7"
          strokeWidth={1}
          strokeLinecap="round"
        />,
      );
      nodes.push(
        <circle
          key={`ab-${i}`}
          cx={bx}
          cy={by}
          r={2.4}
          fill="#e8e8ed"
          stroke="#c8c8cd"
          strokeWidth={0.5}
          className="polymer-atom polymer-atom--branch"
          style={{ animationDelay: `${(i * 0.085 + 0.4) % 2}s` }}
        />,
      );
    }
  }

  return (
    <>
      {lines}
      {nodes}
    </>
  );
}

export function PolymerChainsBackground() {
  const n = 52;
  const vbH = 6 + n * 14 + 24;
  const vbW = 88;

  return (
    <>
      <div className="polymer-rail polymer-rail--left" aria-hidden>
        <div className="polymer-chain-bounce">
          <div className="polymer-chain-sway">
            <svg
              className="polymer-rail-svg"
              viewBox={`0 0 ${vbW} ${vbH}`}
              preserveAspectRatio="xMidYMin slice"
            >
              <g opacity={0.9}>{buildChainElements(n, { cxBase: 44, step: 14, mirrorX: false })}</g>
            </svg>
          </div>
        </div>
      </div>
      <div className="polymer-rail polymer-rail--right" aria-hidden>
        <div className="polymer-chain-bounce polymer-chain-bounce--delayed">
          <div className="polymer-chain-sway polymer-chain-sway--delayed">
            <svg
              className="polymer-rail-svg polymer-rail-svg--flip"
              viewBox={`0 0 ${vbW} ${vbH}`}
              preserveAspectRatio="xMidYMin slice"
            >
              <g opacity={0.9}>{buildChainElements(n, { cxBase: 44, step: 14, mirrorX: true })}</g>
            </svg>
          </div>
        </div>
      </div>
    </>
  );
}
