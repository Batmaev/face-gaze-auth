interface Props {
  stage: "uploading" | "checking";
  progress: number;
}

export default function ProcessingScreen({ stage, progress }: Props) {
  return (
    <div className="screen">
      <div className="content center">
        <div className="spinner" />
        <h2>{stage === "uploading" ? "Видео загружается" : "Идёт проверка"}</h2>
        {stage === "uploading" && (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        )}
      </div>
    </div>
  );
}
