"use client";
/**
 * A button that requires a second click to fire.
 *
 * Used for the irreversible actions in the draft room — recording a pick,
 * wiping a session — where a single stray click during a live draft is
 * expensive and there's no undo worth relying on (undo is disabled outright
 * while Sleeper sync is active).
 *
 * The armed state disarms itself after `timeoutMs` and on blur, so a
 * half-pressed button never sits waiting to fire later, long after you've
 * moved on to something else.
 */

import { ReactNode, useEffect, useState } from "react";

interface Props {
  label: ReactNode;
  confirmLabel?: ReactNode;
  onConfirm: () => void;
  /** Accessible name in the resting state. */
  ariaLabel: string;
  /** Accessible name once armed. Defaults to "Confirm: {ariaLabel}". */
  confirmAriaLabel?: string;
  className?: string;
  confirmClassName?: string;
  disabled?: boolean;
  title?: string;
  timeoutMs?: number;
}

export default function ConfirmButton({
  label,
  confirmLabel = "Confirm?",
  onConfirm,
  ariaLabel,
  confirmAriaLabel,
  className = "",
  confirmClassName = "bg-amber-500 hover:bg-amber-400 text-slate-950",
  disabled = false,
  title,
  timeoutMs = 4000,
}: Props) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), timeoutMs);
    return () => clearTimeout(t);
  }, [armed, timeoutMs]);

  // Disarm whenever the button becomes unavailable, so it can't come back
  // already armed.
  useEffect(() => {
    if (disabled) setArmed(false);
  }, [disabled]);

  if (armed) {
    return (
      <button
        type="button"
        onClick={() => {
          setArmed(false);
          onConfirm();
        }}
        onBlur={() => setArmed(false)}
        onKeyDown={(e) => e.key === "Escape" && setArmed(false)}
        autoFocus
        aria-label={confirmAriaLabel ?? `Confirm: ${ariaLabel}`}
        className={`${className} ${confirmClassName}`}
      >
        {confirmLabel}
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setArmed(true)}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      className={className}
    >
      {label}
    </button>
  );
}
