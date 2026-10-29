import React, { useState, useCallback } from "react";
import "./ImageUpload.css";

interface ImageUploadProps {
  onImageUpload: (_dataUrl: string) => void;
  disabled: boolean;
}

const SUPPORTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE_MB = 5;

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageUpload,
  disabled,
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!SUPPORTED_TYPES.includes(file.type)) {
        alert(`Please upload a JPEG, PNG, or WebP image.`);
        return;
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        alert(`Image must be smaller than ${MAX_SIZE_MB} MB.`);
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        setPreview(dataUrl);
        onImageUpload(dataUrl);
      };
      reader.readAsDataURL(file);
    },
    [onImageUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [disabled, handleFile]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) setDragOver(true);
    },
    [disabled]
  );

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="upload-wrapper">
      <div
        className={`upload-area ${dragOver ? "drag-over" : ""} ${disabled ? "disabled" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleInputChange}
          disabled={disabled}
          id="file-input"
          className="file-input"
        />
        <label htmlFor="file-input" className="upload-button">
          Choose a photo
        </label>
        <p className="upload-hint">
          Drag & drop an image, or click to browse
        </p>
      </div>
      {preview && (
        <img src={preview} alt="Preview" className="upload-preview" />
      )}
    </div>
  );
};

export default ImageUpload;
