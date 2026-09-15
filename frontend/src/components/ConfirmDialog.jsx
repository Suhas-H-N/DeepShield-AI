import React, { useEffect, useRef } from 'react';
import './ConfirmDialog.css';

const ConfirmDialog = ({ open, title, message, confirmLabel = 'Confirm', danger = true, onConfirm, onCancel }) => {
  const confirmRef = useRef(null);

  useEffect(() => {
    if (open && confirmRef.current) {
      confirmRef.current.focus();
    }
  }, [open]);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onCancel();
    };
    if (open) document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div className="confirm-overlay" role="presentation" onClick={onCancel}>
      <div
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 id="confirm-title">{title}</h3>
        <p>{message}</p>
        <div className="confirm-actions">
          <button className="btn btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button
            ref={confirmRef}
            className={danger ? 'btn btn-danger' : 'btn btn-primary'}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;
