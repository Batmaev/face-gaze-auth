import { useCallback, useEffect, useRef, useState } from "react";
import { checkLivenessVideo, getStimulus, type LivenessResult, type StimulusResponse } from "./api";
import { extensionFor, openCamera, type Recording } from "./camera";
import { isMobile, isStandalone } from "./pwa";
import { isPwaSkipped, loadUser, saveUser, skipPwa, type User } from "./storage";
import InstallScreen from "./screens/InstallScreen";
import RegisterScreen from "./screens/RegisterScreen";
import UsersScreen from "./screens/UsersScreen";
import InstructionScreen from "./screens/InstructionScreen";
import StimulusScreen, { type StimulusRun } from "./screens/StimulusScreen";
import ProcessingScreen from "./screens/ProcessingScreen";
import ResultScreen from "./screens/ResultScreen";

type Screen =
  | { name: "install" }
  | { name: "register"; canGoBack: boolean }
  | { name: "users"; canGoBack: boolean }
  | { name: "instruction" }
  | { name: "stimulus"; stimulus: StimulusResponse }
  | { name: "processing"; stage: "uploading" | "checking"; progress: number }
  | { name: "result"; stimulus: StimulusResponse; result: LivenessResult; uploadMs: number; processingMs: number }
  | { name: "error"; message: string };

// Сервер стимулы не хранит, но залежавшийся предзапрошенный стимул лучше заменить свежим.
const STIMULUS_MAX_AGE_MS = 120_000;

function initialScreen(user: User | null): Screen {
  if (isMobile() && !isStandalone() && !isPwaSkipped()) return { name: "install" };
  if (!user) return { name: "register", canGoBack: false };
  return { name: "instruction" };
}

export default function App() {
  const [user, setUser] = useState<User | null>(loadUser);
  const [screen, setScreen] = useState<Screen>(() => initialScreen(user));
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const prefetched = useRef<{ promise: Promise<StimulusResponse>; at: number } | null>(null);

  const cameraPending = useRef(false);

  const requestCamera = useCallback(async () => {
    if (cameraPending.current) return;
    cameraPending.current = true;
    setCameraError(null);
    try {
      const opened = await openCamera();
      opened.getVideoTracks()[0]?.addEventListener("ended", () => setStream((current) => (current === opened ? null : current)));
      setStream(opened);
    } catch (error) {
      setCameraError(errorText(error));
    } finally {
      cameraPending.current = false;
    }
  }, []);

  // Камеру запрашиваем заранее, как только дошли до инструкции, и держим открытой между попытками.
  useEffect(() => {
    if (screen.name === "instruction" && !stream && !cameraError) void requestCamera();
  }, [screen.name, stream, cameraError, requestCamera]);

  useEffect(() => () => stream?.getTracks().forEach((track) => track.stop()), [stream]);

  const prefetchStimulus = useCallback(() => {
    const cached = prefetched.current;
    if (cached && Date.now() - cached.at < STIMULUS_MAX_AGE_MS) return;
    const promise = getStimulus();
    promise.catch(() => {
      if (prefetched.current?.promise === promise) prefetched.current = null;
    });
    prefetched.current = { promise, at: Date.now() };
  }, []);

  useEffect(() => {
    if (screen.name === "instruction") prefetchStimulus();
  }, [screen.name, prefetchStimulus]);

  const selectUser = (selected: User) => {
    saveUser(selected);
    setUser(selected);
    setScreen({ name: "instruction" });
  };

  const startTest = async () => {
    prefetchStimulus();
    const pending = prefetched.current!;
    prefetched.current = null;
    try {
      const stimulus = await pending.promise;
      if (isMobile() && !isStandalone()) document.documentElement.requestFullscreen?.().catch(() => {});
      setScreen({ name: "stimulus", stimulus });
    } catch (error) {
      setScreen({ name: "error", message: `Не удалось получить стимул: ${errorText(error)}` });
    }
  };

  const finishTest = async (stimulus: StimulusResponse, recording: Recording, run: StimulusRun) => {
    const startedAt = performance.now();
    let uploadedAt: number | null = null;
    setScreen({ name: "processing", stage: "uploading", progress: 0 });
    const track = stream?.getVideoTracks()[0];
    try {
      const result = await checkLivenessVideo({
        userId: user!.id,
        stim: stimulus.stim,
        video: recording.blob,
        filename: `video.${extensionFor(recording.mimeType)}`,
        metadata: {
          mime_type: recording.mimeType,
          video_size_bytes: recording.blob.size,
          recording_duration_ms: recording.stoppedAt - recording.startedAt,
          stimulus_start_delay_ms: run.startedAt - recording.startedAt,
          stimulus_duration_ms: run.finishedAt - run.startedAt,
          stimulus_rendered_frames: run.renderedFrames,
          stimulus_fps: stimulus.stim.fps,
          // Все размеры в CSS-пикселях; физические = CSS × device_pixel_ratio.
          stimulus_area: run.area,
          viewport: run.viewport,
          window_position: { x: window.screenX, y: window.screenY },
          screen: {
            width: window.screen.width,
            height: window.screen.height,
            avail_width: window.screen.availWidth,
            avail_height: window.screen.availHeight,
            device_pixel_ratio: window.devicePixelRatio,
            orientation: window.screen.orientation?.type,
          },
          fullscreen: document.fullscreenElement != null,
          mobile: isMobile(),
          camera: track?.getSettings(),
          camera_max_fps: track?.getCapabilities?.().frameRate?.max,
          standalone: isStandalone(),
          user_agent: navigator.userAgent,
        },
        onUploadProgress: (progress) => setScreen({ name: "processing", stage: "uploading", progress }),
        onUploaded: () => {
          uploadedAt = performance.now();
          setScreen({ name: "processing", stage: "checking", progress: 1 });
        },
      });
      const finishedAt = performance.now();
      const uploadEndedAt = uploadedAt ?? finishedAt;
      setScreen({
        name: "result",
        stimulus,
        result,
        uploadMs: uploadEndedAt - startedAt,
        processingMs: finishedAt - uploadEndedAt,
      });
    } catch (error) {
      setScreen({ name: "error", message: `Проверка не удалась: ${errorText(error)}` });
    }
  };

  switch (screen.name) {
    case "install":
      return (
        <InstallScreen
          onSkip={() => {
            skipPwa();
            setScreen(initialScreen(user));
          }}
        />
      );
    case "register":
      return (
        <RegisterScreen
          onBack={screen.canGoBack ? () => setScreen({ name: "instruction" }) : undefined}
          onRegistered={selectUser}
          onShowUsers={() => setScreen({ name: "users", canGoBack: screen.canGoBack })}
        />
      );
    case "users":
      return (
        <UsersScreen
          currentUserId={user?.id}
          onBack={() => setScreen({ name: "register", canGoBack: screen.canGoBack })}
          onSelect={selectUser}
        />
      );
    case "instruction":
      return (
        <InstructionScreen
          user={user!}
          stream={stream}
          cameraError={cameraError}
          onRetryCamera={requestCamera}
          onSettings={() => setScreen({ name: "register", canGoBack: true })}
          onNext={startTest}
        />
      );
    case "stimulus":
      return (
        <StimulusScreen
          stimulus={screen.stimulus}
          stream={stream!}
          onFinished={(recording, run) => finishTest(screen.stimulus, recording, run)}
          onError={(message) => setScreen({ name: "error", message })}
        />
      );
    case "processing":
      return <ProcessingScreen stage={screen.stage} progress={screen.progress} />;
    case "result":
      return (
        <ResultScreen
          stimulus={screen.stimulus}
          result={screen.result}
          uploadMs={screen.uploadMs}
          processingMs={screen.processingMs}
          onRetry={() => setScreen({ name: "instruction" })}
        />
      );
    case "error":
      return (
        <div className="screen">
          <div className="content center">
            <h2>Что-то пошло не так</h2>
            <p className="muted">{screen.message}</p>
          </div>
          <div className="footer">
            <button className="primary" onClick={() => setScreen({ name: "instruction" })}>
              Ещё раз
            </button>
          </div>
        </div>
      );
  }
}

function errorText(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
