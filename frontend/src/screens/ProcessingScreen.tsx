const PROCESSING_TIME_SEC = Number(import.meta.env.VITE_PROCESSING_TIME_SEC ?? 25);

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
        {stage === "uploading" ? (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        ) : (
          <p className="muted">
            Это может занять до {PROCESSING_TIME_SEC}{" "}
            {PROCESSING_TIME_SEC % 10 === 1 && PROCESSING_TIME_SEC % 100 !== 11 ? "секунды" : "секунд"}
          </p>
        )}
      </div>
    </div>
  );
}
