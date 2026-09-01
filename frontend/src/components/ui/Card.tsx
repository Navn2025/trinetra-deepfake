import type { ComponentPropsWithoutRef } from "react";

export function Card({ className = "", ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div
      className={`rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] ${className}`}
      {...rest}
    />
  );
}
