import fsSync from "node:fs";
import fs from "node:fs/promises";
import { execFile } from "node:child_process";
import net from "node:net";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { once } from "node:events";

import makeWASocket, {
  Browsers,
  BufferJSON,
  DisconnectReason,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
  initAuthCreds,
  jidNormalizedUser,
  proto,
} from "baileys";
import pino from "pino";
import qrcodeTerminal from "qrcode-terminal";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..", "..");

const EVENT_ASSISTANT_FINAL = "AssistantFinal";
const EVENT_PERMISSION_REQUEST = "PermissionRequest";
const EVENT_PERMISSION_RESPONSE = "PermissionResponse";
const EVENT_STT_FINAL = "STTFinal";
const EVENT_TURN_PATCH = "TurnPatch";
const EVENT_WAKE = "WakeEvent";
const IMAGE_FILE_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".bmp",
  ".gif",
  ".tif",
  ".tiff",
]);
const VIDEO_FILE_EXTENSIONS = new Set([
  ".mp4",
  ".mov",
  ".avi",
  ".mkv",
  ".webm",
  ".m4v",
  ".3gp",
]);
const TEXT_FILE_EXTENSIONS = new Set([
  ".txt",
  ".md",
  ".rst",
  ".json",
  ".jsonl",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".cfg",
  ".conf",
  ".log",
  ".csv",
  ".tsv",
  ".xml",
  ".html",
  ".css",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".py",
  ".c",
  ".cpp",
  ".cc",
  ".cxx",
  ".h",
  ".hpp",
  ".java",
  ".kt",
  ".go",
  ".rs",
  ".rb",
  ".php",
  ".sh",
  ".ps1",
  ".sql",
]);
const VALID_REQUESTED_MODES = new Set(["COGNITION", "SYSTEM0"]);
const execFileAsync = promisify(execFile);

function makeEvent(eventType, payload) {
  return { event_type: eventType, payload };
}

function loadDotenvDefaults(envPath) {
  let raw = "";
  try {
    raw = fsSync.readFileSync(envPath, "utf8");
  } catch {
    return;
  }
  for (const rawLine of raw.split(/\r?\n/)) {
    let line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    if (line.startsWith("export ")) {
      line = line.slice(7).trim();
    }
    const idx = line.indexOf("=");
    if (idx < 1) {
      continue;
    }
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    if (!key || Object.prototype.hasOwnProperty.call(process.env, key)) {
      continue;
    }
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

function readEnv(name) {
  const value = process.env[name];
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function readEnvAllowEmpty(name) {
  const value = process.env[name];
  if (typeof value !== "string") {
    return null;
  }
  return value.trim();
}

function coerceBool(value, defaultValue) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "off"].includes(normalized)) {
      return false;
    }
  }
  return defaultValue;
}

function coerceInt(value, defaultValue, minimum, maximum) {
  let parsed = defaultValue;
  if (typeof value === "number" && Number.isFinite(value)) {
    parsed = Math.trunc(value);
  } else if (typeof value === "string" && value.trim()) {
    const next = Number.parseInt(value.trim(), 10);
    if (Number.isFinite(next)) {
      parsed = next;
    }
  }
  if (parsed < minimum) {
    return minimum;
  }
  if (parsed > maximum) {
    return maximum;
  }
  return parsed;
}

function parseList(value) {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item ?? "").trim())
      .filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}

function resolvePath(pathValue, baseRoot) {
  const text = String(pathValue || "").trim();
  if (!text) {
    return baseRoot;
  }
  if (path.isAbsolute(text)) {
    return text;
  }
  return path.resolve(baseRoot, text);
}

function normalizePhoneNumber(value) {
  const digits = String(value || "").replace(/\D+/g, "");
  return digits || "";
}

function normalizeChatJid(value) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  if (text.endsWith("@g.us") || text.endsWith("@newsletter") || text.endsWith("@broadcast")) {
    return text;
  }
  if (text.includes("@")) {
    return jidNormalizedUser(text);
  }
  const digits = normalizePhoneNumber(text);
  return digits ? `${digits}@s.whatsapp.net` : text;
}

function fileExtension(fileName) {
  return path.extname(String(fileName || "")).toLowerCase();
}

function normalizeMimeType(value) {
  return String(value || "").trim().toLowerCase();
}

function normalizeRequestedMode(value, defaultValue = "SYSTEM0") {
  const normalized = String(value || "").trim().toUpperCase();
  if (VALID_REQUESTED_MODES.has(normalized)) {
    return normalized;
  }
  return defaultValue;
}

function sanitizeFileName(fileName) {
  const base = path.basename(String(fileName || "").trim());
  if (!base || base === "." || base === "..") {
    return "";
  }
  const sanitized = base.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").replace(/\s+/g, " ").trim();
  return sanitized.slice(0, 120);
}

function coerceByteLength(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value));
  }
  const parsed = Number.parseInt(String(value ?? "").trim(), 10);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }
  return 0;
}

function extensionFromMime(mimeType, fallback = ".bin") {
  switch (normalizeMimeType(mimeType).split(";")[0]) {
    case "image/jpeg":
      return ".jpg";
    case "image/png":
      return ".png";
    case "image/webp":
      return ".webp";
    case "image/gif":
      return ".gif";
    case "image/bmp":
      return ".bmp";
    case "image/tiff":
      return ".tiff";
    case "video/mp4":
      return ".mp4";
    case "video/quicktime":
      return ".mov";
    case "video/webm":
      return ".webm";
    case "video/x-matroska":
      return ".mkv";
    case "video/3gpp":
      return ".3gp";
    case "audio/ogg":
      return ".ogg";
    case "audio/ogg; codecs=opus":
      return ".ogg";
    case "audio/mpeg":
      return ".mp3";
    case "audio/mp4":
    case "audio/aac":
      return ".m4a";
    case "audio/wav":
    case "audio/x-wav":
      return ".wav";
    case "application/pdf":
      return ".pdf";
    case "text/plain":
      return ".txt";
    case "application/json":
      return ".json";
    default:
      return fallback;
  }
}

function suggestFileName(prefix, fileName, mimeType, fallbackExtension = ".bin") {
  const sanitized = sanitizeFileName(fileName);
  if (sanitized) {
    return sanitized;
  }
  return `${prefix}${extensionFromMime(mimeType, fallbackExtension)}`;
}

function isImageAttachment(fileName, mimeType) {
  const mime = normalizeMimeType(mimeType);
  if (mime.startsWith("image/")) {
    return true;
  }
  return IMAGE_FILE_EXTENSIONS.has(fileExtension(fileName));
}

function isVideoAttachment(fileName, mimeType) {
  const mime = normalizeMimeType(mimeType);
  if (mime.startsWith("video/")) {
    return true;
  }
  return VIDEO_FILE_EXTENSIONS.has(fileExtension(fileName));
}

function isPdfAttachment(fileName, mimeType) {
  return normalizeMimeType(mimeType) === "application/pdf" || fileExtension(fileName) === ".pdf";
}

function isTextAttachment(fileName, mimeType) {
  const mime = normalizeMimeType(mimeType);
  if (mime.startsWith("text/")) {
    return true;
  }
  return TEXT_FILE_EXTENSIONS.has(fileExtension(fileName));
}

function buildUnsupportedAttachmentTurnText({
  kind,
  fileName,
  mimeType,
  fileSize,
  fileRef,
  userRequest,
}) {
  return [
    "User sent an unsupported file attachment on WhatsApp.",
    `Attachment kind: ${kind}`,
    `Attachment: ${fileName}`,
    `MIME type: ${mimeType || "unknown"}`,
    `File size bytes: ${fileSize}`,
    `Saved path: ${fileRef}`,
    `User request: ${userRequest}`,
    "Inspect and analyze this file.",
  ].join("\n");
}

function splitMessage(text, limit) {
  if (text.length <= limit) {
    return [text];
  }
  const chunks = [];
  let remaining = text;
  while (remaining) {
    if (remaining.length <= limit) {
      chunks.push(remaining);
      break;
    }
    let cut = remaining.lastIndexOf("\n", limit);
    if (cut < Math.floor(limit / 3)) {
      cut = remaining.lastIndexOf(" ", limit);
    }
    if (cut < Math.floor(limit / 3)) {
      cut = limit;
    }
    const chunk = remaining.slice(0, cut).trim();
    chunks.push(chunk || remaining.slice(0, limit));
    remaining = remaining.slice(cut).trimStart();
  }
  return chunks;
}

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return '"[unserializable]"';
  }
}

function truncateForLog(text, limit = 160) {
  const value = String(text || "");
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, limit)}...`;
}

function messageShapeSummary(message) {
  const root = message?.message && typeof message.message === "object" ? Object.keys(message.message) : [];
  const unwrapped = unwrapMessageContent(message?.message);
  const unwrappedKeys = unwrapped && typeof unwrapped === "object" ? Object.keys(unwrapped) : [];
  return {
    rootKeys: root,
    unwrappedKeys,
  };
}

function unwrapMessageContent(message) {
  let current = message || null;
  while (current && typeof current === "object") {
    if (current.deviceSentMessage?.message) {
      current = current.deviceSentMessage.message;
      continue;
    }
    if (current.ephemeralMessage?.message) {
      current = current.ephemeralMessage.message;
      continue;
    }
    if (current.viewOnceMessage?.message) {
      current = current.viewOnceMessage.message;
      continue;
    }
    if (current.viewOnceMessageV2?.message) {
      current = current.viewOnceMessageV2.message;
      continue;
    }
    if (current.viewOnceMessageV2Extension?.message) {
      current = current.viewOnceMessageV2Extension.message;
      continue;
    }
    if (current.documentWithCaptionMessage?.message) {
      current = current.documentWithCaptionMessage.message;
      continue;
    }
    break;
  }
  return current;
}

function classifyMessage(message) {
  const content = unwrapMessageContent(message?.message);
  if (!content || typeof content !== "object") {
    return { kind: "unknown", text: "" };
  }
  if (typeof content.conversation === "string") {
    return { kind: "text", text: content.conversation };
  }
  if (typeof content.extendedTextMessage?.text === "string") {
    return { kind: "text", text: content.extendedTextMessage.text };
  }
  if (
    content.imageMessage ||
    content.videoMessage ||
    content.audioMessage ||
    content.documentMessage ||
    content.stickerMessage
  ) {
    return { kind: "media", text: "" };
  }
  return { kind: "unknown", text: "" };
}

function extractMediaInfo(message) {
  const content = unwrapMessageContent(message?.message);
  if (!content || typeof content !== "object") {
    return null;
  }
  if (content.imageMessage && typeof content.imageMessage === "object") {
    const payload = content.imageMessage;
    const mimeType = normalizeMimeType(payload.mimetype);
    return {
      kind: "image",
      payload,
      caption: String(payload.caption || "").trim(),
      mimeType,
      fileSize: coerceByteLength(payload.fileLength),
      suggestedName: suggestFileName("image", payload.fileName || payload.filename, mimeType, ".jpg"),
    };
  }
  if (content.videoMessage && typeof content.videoMessage === "object") {
    const payload = content.videoMessage;
    const mimeType = normalizeMimeType(payload.mimetype);
    return {
      kind: "video",
      payload,
      caption: String(payload.caption || "").trim(),
      mimeType,
      fileSize: coerceByteLength(payload.fileLength),
      suggestedName: suggestFileName("video", payload.fileName || payload.filename, mimeType, ".mp4"),
    };
  }
  if (content.audioMessage && typeof content.audioMessage === "object") {
    const payload = content.audioMessage;
    const mimeType = normalizeMimeType(payload.mimetype);
    const prefix = payload.ptt ? "voice" : "audio";
    const fallback = payload.ptt ? ".ogg" : ".bin";
    return {
      kind: "audio",
      payload,
      caption: "",
      mimeType,
      fileSize: coerceByteLength(payload.fileLength),
      suggestedName: suggestFileName(prefix, payload.fileName || payload.filename, mimeType, fallback),
    };
  }
  if (content.documentMessage && typeof content.documentMessage === "object") {
    const payload = content.documentMessage;
    const mimeType = normalizeMimeType(payload.mimetype);
    return {
      kind: "document",
      payload,
      caption: String(payload.caption || "").trim(),
      mimeType,
      fileSize: coerceByteLength(payload.fileLength),
      suggestedName: suggestFileName(
        "document",
        payload.fileName || payload.filename,
        mimeType,
        ".bin",
      ),
    };
  }
  if (content.stickerMessage && typeof content.stickerMessage === "object") {
    const payload = content.stickerMessage;
    return {
      kind: "sticker",
      payload,
      caption: "",
      mimeType: "image/webp",
      fileSize: coerceByteLength(payload.fileLength),
      suggestedName: suggestFileName("sticker", payload.fileName || payload.filename, "image/webp", ".webp"),
    };
  }
  return null;
}

async function loadSettings() {
  const settingsPath = path.join(REPO_ROOT, "config", "settings.json");
  const raw = JSON.parse(await fs.readFile(settingsPath, "utf8"));
  const workspaceRoot = resolvePath(raw.workspace_root || ".", REPO_ROOT);
  const dataDir = resolvePath(raw.data_dir || "./data", REPO_ROOT);
  return {
    repoRoot: REPO_ROOT,
    workspaceRoot,
    dataDir,
    rpc: raw.rpc || {},
    interface: raw.interface || {},
    orchestrator: raw.orchestrator || {},
    voice: raw.voice || {},
    video: raw.video || {},
    whatsapp: raw.whatsapp || {},
  };
}

async function createFileAuthState(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  let persisted = {};
  try {
    const raw = await fs.readFile(filePath, "utf8");
    persisted = JSON.parse(raw, BufferJSON.reviver);
  } catch {
    persisted = {};
  }

  const store = {
    creds: persisted.creds || initAuthCreds(),
    keys: persisted.keys && typeof persisted.keys === "object" ? persisted.keys : {},
  };
  let writeChain = Promise.resolve();

  const persist = async () => {
    const snapshot = {
      creds: store.creds,
      keys: store.keys,
    };
    const serialized = JSON.stringify(snapshot, BufferJSON.replacer);
    const tempPath = `${filePath}.tmp`;
    writeChain = writeChain.then(async () => {
      await fs.writeFile(tempPath, serialized, "utf8");
      await fs.rename(tempPath, filePath);
    });
    return writeChain;
  };

  return {
    meta: {
      registered: Boolean(store.creds.registered),
      hasIdentity: Boolean(store.creds.me?.id),
    },
    state: {
      creds: store.creds,
      keys: {
        get: async (type, ids) => {
          const bucket = store.keys[type] || {};
          const data = {};
          for (const id of ids) {
            let value = bucket[id];
            if (type === "app-state-sync-key" && value) {
              value = proto.Message.AppStateSyncKeyData.fromObject(value);
            }
            data[id] = value;
          }
          return data;
        },
        set: async (data) => {
          for (const [category, values] of Object.entries(data || {})) {
            if (!store.keys[category]) {
              store.keys[category] = {};
            }
            for (const [id, value] of Object.entries(values || {})) {
              if (value) {
                store.keys[category][id] = value;
              } else {
                delete store.keys[category][id];
              }
            }
          }
          await persist();
        },
      },
    },
    saveCreds: async () => {
      await persist();
    },
  };
}

class PermissionResolution {
  constructor(requestId, status) {
    this.requestId = requestId;
    this.status = status;
  }

  get matched() {
    return Boolean(this.requestId) && this.status === "matched";
  }
}

class JsonlSession {
  constructor(host, port, callbacks = {}, options = {}) {
    this._host = host;
    this._port = port;
    this._onAssistantFinal = callbacks.onAssistantFinal || null;
    this._onPermissionRequest = callbacks.onPermissionRequest || null;
    this._onClosed = callbacks.onClosed || null;
    this._sourceId = typeof options.sourceId === "string" && options.sourceId.trim()
      ? options.sourceId.trim()
      : null;
    this._socket = null;
    this._reader = null;
    this._writeChain = Promise.resolve();
    this._pendingPermissions = new Map();
    this._lastTurnId = null;
    this._closing = false;
  }

  get lastTurnId() {
    return this._lastTurnId;
  }

  get pendingPermissions() {
    return new Map(this._pendingPermissions);
  }

  async connect() {
    if (this._socket) {
      throw new Error("Session is already connected.");
    }
    const socket = await new Promise((resolve, reject) => {
      const candidate = net.createConnection({ host: this._host, port: this._port });
      const cleanup = () => {
        candidate.off("connect", onConnect);
        candidate.off("error", onError);
      };
      const onConnect = () => {
        cleanup();
        resolve(candidate);
      };
      const onError = (error) => {
        cleanup();
        reject(error);
      };
      candidate.once("connect", onConnect);
      candidate.once("error", onError);
    });
    this._socket = socket;
    this._reader = readline.createInterface({ input: socket, crlfDelay: Infinity });
    this._reader.on("line", (line) => {
      void this._handleLine(line);
    });
    socket.on("error", (error) => {
      console.error(`[whatsapp] orchestrator socket error: ${error.message}`);
    });
    socket.on("close", () => {
      if (!this._closing && this._onClosed) {
        void this._onClosed();
      }
    });
  }

  async close() {
    this._closing = true;
    if (this._reader) {
      this._reader.close();
      this._reader = null;
    }
    const socket = this._socket;
    this._socket = null;
    if (!socket) {
      return;
    }
    socket.end();
    await once(socket, "close").catch(() => null);
  }

  async _handleLine(line) {
    let event;
    try {
      event = JSON.parse(line);
    } catch (error) {
      console.error("[whatsapp] Failed to parse orchestrator JSONL event.");
      return;
    }
    const eventType = event?.event_type;
    const payload = event?.payload && typeof event.payload === "object" ? event.payload : {};

    if (eventType === EVENT_PERMISSION_REQUEST) {
      const requestId = typeof payload.request_id === "string" ? payload.request_id : "";
      if (requestId) {
        this._pendingPermissions.set(requestId, payload);
      }
      if (this._onPermissionRequest) {
        await this._onPermissionRequest(payload);
      }
      return;
    }

    if (eventType === EVENT_ASSISTANT_FINAL && this._onAssistantFinal) {
      await this._onAssistantFinal(payload);
    }
  }

  async sendEvent(event) {
    if (!this._socket) {
      throw new Error("Session is not connected.");
    }
    let outgoing = event;
    if (
      this._sourceId &&
      event &&
      typeof event === "object" &&
      event.payload &&
      typeof event.payload === "object" &&
      !Object.prototype.hasOwnProperty.call(event.payload, "source_id")
    ) {
      outgoing = {
        ...event,
        payload: {
          ...event.payload,
          source_id: this._sourceId,
        },
      };
    }
    const payload = `${JSON.stringify(outgoing)}\n`;
    this._writeChain = this._writeChain.then(
      () =>
        new Promise((resolve, reject) => {
          const socket = this._socket;
          if (!socket) {
            reject(new Error("Session is not connected."));
            return;
          }
          socket.write(payload, "utf8", (error) => {
            if (error) {
              reject(error);
              return;
            }
            resolve();
          });
        }),
    );
    return this._writeChain;
  }

  async sendWake(payload = {}) {
    const wakePayload = { ...payload };
    if (typeof wakePayload.ts !== "number") {
      wakePayload.ts = Date.now() / 1000;
    }
    await this.sendEvent(makeEvent(EVENT_WAKE, wakePayload));
  }

  async submitTurn(text, requestedMode = null, turnId = null, skipDistillation = false) {
    const resolvedTurnId = String(turnId || cryptoRandomId());
    const now = Date.now() / 1000;
    const payload = {
      turn_id: resolvedTurnId,
      text,
      t0: now,
      t1: now,
      is_final: true,
    };
    const normalizedMode = String(requestedMode || "").trim().toUpperCase();
    if (VALID_REQUESTED_MODES.has(normalizedMode)) {
      payload.requested_mode = normalizedMode;
    }
    if (skipDistillation) {
      payload.skip_distillation = true;
    }
    this._lastTurnId = resolvedTurnId;
    await this.sendEvent(makeEvent(EVENT_STT_FINAL, payload));
    return resolvedTurnId;
  }

  async patchTurn(turnId, appendedText) {
    await this.sendEvent(
      makeEvent(EVENT_TURN_PATCH, {
        turn_id: turnId,
        appended_text: appendedText,
      }),
    );
  }

  resolvePermissionId(token = null) {
    if (token) {
      const matches = [...this._pendingPermissions.keys()].filter((requestId) =>
        requestId.startsWith(token),
      );
      if (matches.length === 1) {
        return new PermissionResolution(matches[0], "matched");
      }
      if (matches.length > 1) {
        return new PermissionResolution(null, "ambiguous");
      }
    }
    if (this._pendingPermissions.size === 1) {
      return new PermissionResolution([...this._pendingPermissions.keys()][0], "matched");
    }
    if (this._pendingPermissions.size > 1) {
      return new PermissionResolution(null, "multiple_pending");
    }
    return new PermissionResolution(null, "missing");
  }

  async sendPermissionDecision(requestId, approved) {
    if (!this._pendingPermissions.has(requestId)) {
      return false;
    }
    this._pendingPermissions.delete(requestId);
    const payload = {
      request_id: requestId,
      approved,
    };
    if (approved) {
      payload.token = cryptoRandomId();
    }
    await this.sendEvent(makeEvent(EVENT_PERMISSION_RESPONSE, payload));
    return true;
  }
}

class WhatsAppDaemon {
  constructor(settings) {
    this._settings = settings;
    const cfg = settings.whatsapp || {};
    this._host = settings.rpc?.orch_host || "127.0.0.1";
    this._port = settings.rpc?.orch_port || 50051;
    this._modelHost = settings.rpc?.orch_host || "127.0.0.1";
    this._modelPort = settings.rpc?.model_port || 50054;
    this._cfg = cfg;
    this._authDir = resolvePath(
      readEnv("WHATSAPP_AUTH_DIR") || cfg.auth_dir || path.join(settings.dataDir, "whatsapp_auth"),
      settings.repoRoot,
    );
    this._incomingDir = resolvePath(
      readEnv("WHATSAPP_INCOMING_DIR") || cfg.incoming_dir || path.join(settings.dataDir, "whatsapp_inbox"),
      settings.repoRoot,
    );
    this._sessionName =
      String(readEnv("WHATSAPP_SESSION_NAME") || cfg.session_name || "primary").trim() || "primary";
    const triggerPrefixEnv = readEnvAllowEmpty("WHATSAPP_TRIGGER_PREFIX");
    if (triggerPrefixEnv !== null) {
      this._triggerPrefix = triggerPrefixEnv;
    } else if (Object.prototype.hasOwnProperty.call(cfg, "trigger_prefix")) {
      this._triggerPrefix = String(cfg.trigger_prefix ?? "").trim();
    } else {
      this._triggerPrefix = "/agi";
    }
    this._pairingPhoneNumber = normalizePhoneNumber(
      readEnv("WHATSAPP_PAIRING_PHONE_NUMBER") || cfg.pairing_phone_number || "",
    );
    this._selfChatOnly = coerceBool(
      readEnv("WHATSAPP_SELF_CHAT_ONLY") ?? cfg.self_chat_only,
      true,
    );
    this._allowedChatJids = parseList(
      readEnv("WHATSAPP_ALLOWED_CHAT_JIDS") ?? cfg.allowed_chat_jids,
    ).map((value) => normalizeChatJid(value));
    this._qrInTerminal = coerceBool(readEnv("WHATSAPP_QR_IN_TERMINAL") ?? cfg.qr_in_terminal, true);
    this._sendPresenceUpdates = coerceBool(
      readEnv("WHATSAPP_SEND_PRESENCE_UPDATES") ?? cfg.send_presence_updates,
      true,
    );
    this._sendUserUpdates = coerceBool(
      readEnv("WHATSAPP_SEND_USER_UPDATES") ?? cfg.send_user_updates,
      true,
    );
    this._debugInbound = coerceBool(
      readEnv("WHATSAPP_DEBUG_INBOUND") ?? cfg.debug_inbound,
      false,
    );
    this._maxMessageChars = coerceInt(
      readEnv("WHATSAPP_MAX_MESSAGE_CHARS") || cfg.max_message_chars,
      3500,
      200,
      65000,
    );
    let responseMode = String(
      readEnv("WHATSAPP_RESPONSE_MODE") || cfg.response_mode || "text",
    ).trim().toLowerCase();
    if (!["text", "audio", "same"].includes(responseMode)) {
      responseMode = "text";
    }
    this._responseMode = responseMode;
    this._audioResponseMaxChars = coerceInt(
      readEnv("WHATSAPP_AUDIO_RESPONSE_MAX_CHARS") || cfg.audio_response_max_chars,
      1800,
      128,
      20000,
    );
    const responseTtsRateRaw =
      readEnv("WHATSAPP_TTS_RATE") ?? cfg.tts_rate ?? settings.voice?.tts_rate;
    this._responseTtsRate = coerceInt(
      responseTtsRateRaw,
      175,
      50,
      400,
    );
    const responseTtsVolumeRaw =
      readEnv("WHATSAPP_TTS_VOLUME") ?? cfg.tts_volume ?? settings.voice?.tts_volume ?? "1.0";
    this._responseTtsVolume = Number.parseFloat(String(responseTtsVolumeRaw));
    if (!Number.isFinite(this._responseTtsVolume)) {
      this._responseTtsVolume = 1.0;
    }
    this._responseTtsVolume = Math.max(0.0, Math.min(1.0, this._responseTtsVolume));
    const ffmpegBinaryRaw =
      readEnv("WHATSAPP_FFMPEG_BINARY") ?? cfg.ffmpeg_binary ?? settings.video?.ffmpeg_binary ?? "ffmpeg";
    this._ffmpegBinary = String(
      ffmpegBinaryRaw,
    ).trim() || "ffmpeg";
    this._modelRpcTimeoutS = coerceInt(
      readEnv("WHATSAPP_MODEL_RPC_TIMEOUT_S") ??
        cfg.model_rpc_timeout_s ??
        settings.orchestrator?.model_rpc_timeout_s,
      120,
      0,
      7 * 24 * 60 * 60,
    );
    this._maxDownloadBytes = coerceInt(
      readEnv("WHATSAPP_MAX_DOWNLOAD_BYTES") || cfg.max_download_bytes,
      25 * 1024 * 1024,
      32 * 1024,
      250 * 1024 * 1024,
    );
    this._textAttachmentMaxChars = coerceInt(
      readEnv("WHATSAPP_TEXT_ATTACHMENT_MAX_CHARS") || cfg.text_attachment_max_chars,
      12000,
      512,
      200000,
    );
    this._visionQuestionFallback = String(
      readEnv("WHATSAPP_VISION_QUESTION_FALLBACK") ||
        cfg.vision_question_fallback ||
        "Describe this image.",
    ).trim() || "Describe this image.";
    this._mediaSubmitRequestedMode = normalizeRequestedMode(
      readEnv("WHATSAPP_MEDIA_SUBMIT_REQUESTED_MODE") || cfg.media_submit_requested_mode,
      "SYSTEM0",
    );
    this._mediaSkipDistillation = coerceBool(
      readEnv("WHATSAPP_MEDIA_SKIP_DISTILLATION") ?? cfg.media_skip_distillation,
      false,
    );
    this._pdfMaxPages = coerceInt(
      readEnv("WHATSAPP_PDF_MAX_PAGES") || cfg.pdf_max_pages,
      40,
      1,
      1000,
    );
    this._pdfMaxChars = coerceInt(
      readEnv("WHATSAPP_PDF_MAX_CHARS") || cfg.pdf_max_chars,
      20000,
      512,
      500000,
    );
    this._unsupportedFilesToAgent = coerceBool(
      readEnv("WHATSAPP_UNSUPPORTED_FILES_TO_AGENT") ?? cfg.unsupported_files_to_agent,
      true,
    );
    this._sttModelName = String(
      readEnv("WHATSAPP_STT_MODEL") ||
        cfg.stt_model ||
        settings.voice?.stt_model ||
        "small.en",
    ).trim() || "small.en";
    this._sttDevice = String(
      readEnv("WHATSAPP_STT_DEVICE") ||
        cfg.stt_device ||
        settings.voice?.stt_device ||
        "cpu",
    ).trim() || "cpu";
    this._sttComputeType = String(
      readEnv("WHATSAPP_STT_COMPUTE_TYPE") ||
        cfg.stt_compute_type ||
        settings.voice?.stt_compute_type ||
        "int8",
    ).trim() || "int8";
    this._pythonBin = String(readEnv("WHATSAPP_PYTHON_BIN") || cfg.python_bin || "").trim();
    this._mediaHelperPath = path.join(settings.repoRoot, "src", "whatsapp_daemon", "media_helper.py");
    this._logger = pino({ level: "silent" });
    this._session = new JsonlSession(
      this._host,
      this._port,
      {
        onAssistantFinal: async (payload) => this._handleAssistantFinal(payload),
        onPermissionRequest: async (payload) => this._handlePermissionRequest(payload),
        onClosed: async () => this._handleSessionClosed(),
      },
      { sourceId: "whatsapp" },
    );
    this._sock = null;
    this._selfJid = "";
    this._selfChatAliases = new Set();
    this._activeChatJid = "";
    this._turnChat = new Map();
    this._turnInputMode = new Map();
    this._permissionChat = new Map();
    this._sentMessageIds = new Map();
    this._pairingCodeRequested = false;
    this._stopped = false;
    this._startedAtEpochSeconds = Math.floor(Date.now() / 1000);
    this._resolveRun = null;
    this._rejectRun = null;
    this._authMeta = { registered: false, hasIdentity: false };
  }

  async run() {
    await fs.mkdir(this._authDir, { recursive: true });
    await fs.mkdir(this._incomingDir, { recursive: true });
    await this._session.connect();
    console.log(`[whatsapp] connected to orchestrator at ${this._host}:${this._port}`);
    console.log(`[whatsapp] auth dir: ${this._authDir}`);
    console.log(
      `[whatsapp] mode: ${this._selfChatOnly ? "self-chat only" : "allowed chats / unrestricted"}`,
    );
    if (this._debugInbound) {
      console.log("[whatsapp] inbound debug logging is enabled.");
    }
    await this._connectSocket();
    await new Promise((resolve, reject) => {
      this._resolveRun = resolve;
      this._rejectRun = reject;
    });
  }

  async _connectSocket() {
    const authPath = path.join(this._authDir, `${this._sessionName}.json`);
    const { meta, state, saveCreds } = await createFileAuthState(authPath);
    this._authMeta = meta;
    const versionInfo = await fetchLatestBaileysVersion().catch(() => null);
    const version = versionInfo?.version;

    this._pairingCodeRequested = false;
    if (meta.hasIdentity && !meta.registered) {
      console.log(
        `[whatsapp] found partial auth state in ${authPath}. If login fails immediately, delete this file and retry.`,
      );
    }

    const sock = makeWASocket({
      auth: state,
      browser: Browsers.macOS("AGI"),
      logger: this._logger,
      markOnlineOnConnect: false,
      syncFullHistory: false,
      version,
    });

    this._sock = sock;
    sock.ev.on("creds.update", saveCreds);
    sock.ev.on("connection.update", (update) => {
      void this._handleConnectionUpdate(sock, update);
    });
    sock.ev.on("messages.upsert", (payload) => {
      void this._handleMessagesUpsert(sock, payload);
    });
  }

  async _handleSessionClosed() {
    if (this._stopped) {
      return;
    }
    this._stopped = true;
    if (this._rejectRun) {
      this._rejectRun(new Error("Orchestrator connection closed."));
    }
  }

  async _handleConnectionUpdate(sock, update) {
    if (sock !== this._sock || this._stopped) {
      return;
    }

    const qr = typeof update?.qr === "string" ? update.qr : "";
    if (qr && this._qrInTerminal && !this._pairingPhoneNumber) {
      console.log("[whatsapp] scan the QR code below with WhatsApp Linked Devices:");
      qrcodeTerminal.generate(qr, { small: true });
    }

    if (this._pairingPhoneNumber && !this._pairingCodeRequested && qr) {
      this._pairingCodeRequested = true;
      try {
        const code = await sock.requestPairingCode(this._pairingPhoneNumber);
        console.log(`[whatsapp] pairing code for ${this._pairingPhoneNumber}: ${code}`);
      } catch (error) {
        this._pairingCodeRequested = false;
        console.error(`[whatsapp] Failed to request pairing code: ${error.message}`);
      }
    }

    if (update?.connection === "open") {
      this._selfJid = normalizeChatJid(sock.user?.id || "");
      this._selfChatAliases = new Set(this._selfJid ? [this._selfJid] : []);
      if (this._selfChatOnly) {
        this._activeChatJid = this._selfJid;
      }
      console.log(`[whatsapp] connected as ${this._selfJid || "unknown-user"}`);
      if (this._selfChatOnly) {
        console.log(
          `[whatsapp] self-chat mode is active; send "${this._triggerPrefix} your message" to submit a turn.`,
        );
      }
      return;
    }

    if (update?.connection !== "close") {
      return;
    }

    const statusCode = update?.lastDisconnect?.error?.output?.statusCode;
    const reasonText =
      update?.lastDisconnect?.error?.message ||
      update?.lastDisconnect?.error?.output?.payload?.message ||
      "unknown";
    console.log(
      `[whatsapp] connection closed (status=${statusCode ?? "unknown"}, reason=${reasonText}).`,
    );
    if (statusCode === DisconnectReason.loggedOut) {
      this._stopped = true;
      console.error("[whatsapp] session was logged out. Delete the saved auth file and pair again.");
      if (this._authMeta.hasIdentity && !this._authMeta.registered) {
        console.error(
          `[whatsapp] the saved auth state looks incomplete. Remove ${path.join(this._authDir, `${this._sessionName}.json`)} before retrying.`,
        );
      }
      await this._session.close().catch(() => null);
      if (this._rejectRun) {
        this._rejectRun(new Error("WhatsApp session logged out."));
      }
      return;
    }

    console.log("[whatsapp] connection closed; reconnecting...");
    await delay(statusCode === DisconnectReason.restartRequired ? 250 : 1500);
    if (!this._stopped && sock === this._sock) {
      await this._connectSocket();
    }
  }

  async _handleMessagesUpsert(sock, payload) {
    if (sock !== this._sock || this._stopped) {
      return;
    }
    if (!["notify", "append"].includes(payload?.type) || !Array.isArray(payload?.messages)) {
      return;
    }
    this._debug(
      `messages.upsert type=${payload.type} count=${payload.messages.length} requestId=${payload.requestId || ""}`,
    );
    this._pruneSentMessageIds();
    for (const message of payload.messages) {
      await this._handleIncomingMessage(message);
    }
  }

  async _handleIncomingMessage(message) {
    const key = message?.key || {};
    const remoteJid = normalizeChatJid(key.remoteJid || "");
    const remoteJidAlt = normalizeChatJid(key.remoteJidAlt || "");
    this._maybeLearnSelfChatAlias(remoteJid, remoteJidAlt);
    const shape = messageShapeSummary(message);
    const classified = classifyMessage(message);
    const mediaInfo = extractMediaInfo(message);
    this._debug(
      `incoming id=${key.id || ""} remoteJid=${remoteJid || ""} remoteJidAlt=${remoteJidAlt || ""} fromMe=${Boolean(key.fromMe)} participant=${key.participant || ""} timestamp=${message?.messageTimestamp || ""} kind=${classified.kind} rootKeys=${safeJson(shape.rootKeys)} unwrappedKeys=${safeJson(shape.unwrappedKeys)} text=${safeJson(truncateForLog(classified.text))}`,
    );
    if (mediaInfo) {
      this._debug(
        `media id=${key.id || ""} mediaKind=${mediaInfo.kind} mime=${mediaInfo.mimeType || ""} fileName=${safeJson(mediaInfo.suggestedName)} fileSize=${mediaInfo.fileSize}`,
      );
    }
    if (!remoteJid || this._shouldIgnoreJid(remoteJid)) {
      this._debug(`drop id=${key.id || ""} reason=ignored_jid remoteJid=${remoteJid || ""}`);
      return;
    }
    const timestamp = Number(message?.messageTimestamp || 0);
    if (Number.isFinite(timestamp) && timestamp > 0 && timestamp < this._startedAtEpochSeconds) {
      this._debug(
        `drop id=${key.id || ""} reason=old_message timestamp=${timestamp} startedAt=${this._startedAtEpochSeconds}`,
      );
      return;
    }
    if (this._isSentMessageEcho(key.id)) {
      this._debug(`drop id=${key.id || ""} reason=sent_echo`);
      return;
    }
    if (!this._chatAllowed(remoteJid, remoteJidAlt)) {
      this._debug(
        `drop id=${key.id || ""} reason=chat_not_allowed remoteJid=${remoteJid} remoteJidAlt=${remoteJidAlt} selfJid=${this._selfJid} aliases=${safeJson([...this._selfChatAliases])}`,
      );
      return;
    }

    this._activeChatJid = remoteJid;

    if (classified.kind === "text") {
      this._debug(`accept id=${key.id || ""} action=handle_text`);
      await this._handleIncomingText(remoteJid, classified.text);
      return;
    }
    if (mediaInfo) {
      this._debug(`accept id=${key.id || ""} action=handle_media mediaKind=${mediaInfo.kind}`);
      await this._handleIncomingMedia(message, remoteJid, mediaInfo);
      return;
    }
    this._debug(`drop id=${key.id || ""} reason=unclassified_message`);
  }

  _shouldIgnoreJid(jid) {
    return jid === "status@broadcast" || jid.endsWith("@newsletter");
  }

  _chatAllowed(jid, altJid = "") {
    if (this._selfChatOnly) {
      if (!this._selfJid) {
        return false;
      }
      if (this._selfChatAliases.has(jid) || this._selfChatAliases.has(altJid)) {
        return true;
      }
      return altJid === this._selfJid;
    }
    if (this._allowedChatJids.length > 0) {
      return this._allowedChatJids.includes(jid) || this._allowedChatJids.includes(altJid);
    }
    return true;
  }

  _maybeLearnSelfChatAlias(jid, altJid) {
    if (!this._selfJid || !jid || !altJid) {
      return;
    }
    if (altJid !== this._selfJid || this._selfChatAliases.has(jid)) {
      return;
    }
    this._selfChatAliases.add(jid);
    console.log(`[whatsapp] learned self-chat alias ${jid} for ${this._selfJid}`);
  }

  _isSentMessageEcho(messageId) {
    if (!messageId) {
      return false;
    }
    return this._sentMessageIds.has(messageId);
  }

  _rememberSentMessage(messageId) {
    if (!messageId) {
      return;
    }
    this._sentMessageIds.set(messageId, Date.now());
  }

  _pruneSentMessageIds() {
    const cutoff = Date.now() - 10 * 60 * 1000;
    for (const [messageId, ts] of this._sentMessageIds.entries()) {
      if (ts < cutoff) {
        this._sentMessageIds.delete(messageId);
      }
    }
  }

  async _handleIncomingText(chatJid, rawText) {
    const line = String(rawText || "").trim();
    if (!line) {
      this._debug(`drop_text reason=empty chatJid=${chatJid}`);
      return;
    }
    this._debug(`text chatJid=${chatJid} line=${safeJson(line)}`);
    if (line === "/help") {
      await this._sendHelp(chatJid);
      return;
    }
    if (line === "/status") {
      await this._sendStatus(chatJid);
      return;
    }
    if (line === "/wake") {
      await this._session.sendWake();
      await this._sendMessage(chatJid, "Wake event sent.");
      return;
    }
    if (line.startsWith("/approve") || line.startsWith("/deny")) {
      await this._handlePermissionCommand(chatJid, line);
      return;
    }
    if (line.startsWith("+")) {
      await this._handlePatch(chatJid, line);
      return;
    }
    if (line.startsWith("/")) {
      if (this._triggerPrefix && line.startsWith(this._triggerPrefix)) {
        const prompt = line.slice(this._triggerPrefix.length).trim();
        if (!prompt) {
          await this._sendMessage(chatJid, `Message after ${this._triggerPrefix} is empty.`);
          return;
        }
        await this._submitTurn(chatJid, prompt);
        return;
      }
      await this._sendHelp(chatJid);
      return;
    }
    if (!this._triggerPrefix) {
      await this._submitTurn(chatJid, line);
      return;
    }
    this._debug(
      `drop_text reason=missing_trigger_prefix expected=${safeJson(this._triggerPrefix)} line=${safeJson(line)}`,
    );
  }

  async _handleIncomingMedia(message, chatJid, mediaInfo) {
    if (mediaInfo.kind === "audio") {
      await this._handleAudioAttachment(message, chatJid, mediaInfo);
      return;
    }
    if (mediaInfo.kind === "image") {
      await this._handleImageAttachment(message, chatJid, mediaInfo);
      return;
    }
    if (mediaInfo.kind === "video") {
      await this._handleVideoAttachment(message, chatJid, mediaInfo);
      return;
    }
    if (mediaInfo.kind === "document") {
      await this._handleDocumentAttachment(message, chatJid, mediaInfo);
      return;
    }
    if (this._unsupportedFilesToAgent) {
      await this._handleUnsupportedAttachment(message, chatJid, mediaInfo);
      return;
    }
    await this._sendMessage(
      chatJid,
      "Unsupported file type. Send audio, images, videos, PDFs, or text/code files for now.",
    );
  }

  async _handleAudioAttachment(message, chatJid, mediaInfo) {
    await this._sendUserUpdate(chatJid, "Transcribing audio...");
    let download;
    let helperResult;
    try {
      download = await this._downloadMediaToFile(message, "audio", mediaInfo.suggestedName);
      helperResult = await this._invokeMediaHelper("audio", [
        "--file",
        download.localPath,
        "--caption",
        mediaInfo.caption || "",
        "--platform",
        "WhatsApp",
        "--stt-model",
        this._sttModelName,
        "--stt-device",
        this._sttDevice,
        "--stt-compute-type",
        this._sttComputeType,
      ]);
    } catch (error) {
      await this._sendMessage(chatJid, `Failed to transcribe audio: ${error.message}`);
      return;
    }
    if (helperResult?.ok !== true) {
      await this._sendMessage(chatJid, helperResult?.user_message || "Failed to transcribe audio.");
      return;
    }
    await this._submitTurn(chatJid, helperResult.turn_text || "", "audio");
  }

  async _handleImageAttachment(message, chatJid, mediaInfo) {
    const question = mediaInfo.caption || this._visionQuestionFallback;
    await this._sendUserUpdate(chatJid, "Analyzing image...");
    let download;
    let helperResult;
    try {
      download = await this._downloadMediaToFile(message, "image", mediaInfo.suggestedName);
      helperResult = await this._invokeMediaHelper("image", [
        "--file",
        download.localPath,
        "--display-name",
        mediaInfo.suggestedName,
        "--question",
        question,
        "--platform",
        "WhatsApp",
        "--model-host",
        this._modelHost,
        "--model-port",
        String(this._modelPort),
        "--timeout-s",
        String(this._modelRpcTimeoutS),
      ]);
    } catch (error) {
      await this._sendMessage(chatJid, `Failed to process image: ${error.message}`);
      return;
    }
    if (helperResult?.ok !== true) {
      await this._sendMessage(chatJid, helperResult?.user_message || "Failed to process image.");
      return;
    }
    await this._submitTurn(
      chatJid,
      helperResult.turn_text || "",
      "text",
      this._mediaSubmitRequestedMode,
      this._mediaSkipDistillation,
    );
  }

  async _handleVideoAttachment(message, chatJid, mediaInfo) {
    const question = mediaInfo.caption || `Please analyze this video (${mediaInfo.suggestedName}).`;
    await this._sendUserUpdate(chatJid, "Analyzing video...");
    let download;
    let helperResult;
    try {
      download = await this._downloadMediaToFile(message, "video", mediaInfo.suggestedName);
      helperResult = await this._invokeMediaHelper("video", [
        "--file",
        download.localPath,
        "--display-name",
        mediaInfo.suggestedName,
        "--question",
        question,
        "--platform",
        "WhatsApp",
        "--model-host",
        this._modelHost,
        "--model-port",
        String(this._modelPort),
        "--timeout-s",
        String(this._modelRpcTimeoutS),
      ]);
    } catch (error) {
      await this._sendMessage(chatJid, `Failed to process video: ${error.message}`);
      return;
    }
    if (helperResult?.ok !== true) {
      await this._sendMessage(chatJid, helperResult?.user_message || "Failed to process video.");
      return;
    }
    await this._submitTurn(
      chatJid,
      helperResult.turn_text || "",
      "text",
      this._mediaSubmitRequestedMode,
      this._mediaSkipDistillation,
    );
  }

  async _handleDocumentAttachment(message, chatJid, mediaInfo) {
    if (isImageAttachment(mediaInfo.suggestedName, mediaInfo.mimeType)) {
      await this._handleImageAttachment(message, chatJid, {
        ...mediaInfo,
        kind: "image",
      });
      return;
    }
    if (isVideoAttachment(mediaInfo.suggestedName, mediaInfo.mimeType)) {
      await this._handleVideoAttachment(message, chatJid, {
        ...mediaInfo,
        kind: "video",
      });
      return;
    }
    if (isPdfAttachment(mediaInfo.suggestedName, mediaInfo.mimeType)) {
      await this._sendUserUpdate(chatJid, "Reading PDF...");
      let download;
      let helperResult;
      try {
        download = await this._downloadMediaToFile(message, "file", mediaInfo.suggestedName);
        helperResult = await this._invokeMediaHelper("pdf", [
          "--file",
          download.localPath,
          "--display-name",
          mediaInfo.suggestedName,
          "--caption",
          mediaInfo.caption || "",
          "--platform",
          "WhatsApp",
          "--max-pages",
          String(this._pdfMaxPages),
          "--max-chars",
          String(this._pdfMaxChars),
        ]);
      } catch (error) {
        await this._sendMessage(chatJid, `Failed to read PDF: ${error.message}`);
        return;
      }
      if (helperResult?.ok !== true) {
        await this._sendMessage(chatJid, helperResult?.user_message || "Failed to read PDF.");
        return;
      }
      await this._submitTurn(chatJid, helperResult.turn_text || "", "text");
      return;
    }
    if (isTextAttachment(mediaInfo.suggestedName, mediaInfo.mimeType)) {
      await this._sendUserUpdate(chatJid, "Reading file...");
      let download;
      let helperResult;
      try {
        download = await this._downloadMediaToFile(message, "file", mediaInfo.suggestedName);
        helperResult = await this._invokeMediaHelper("text", [
          "--file",
          download.localPath,
          "--display-name",
          mediaInfo.suggestedName,
          "--caption",
          mediaInfo.caption || "",
          "--platform",
          "WhatsApp",
          "--max-chars",
          String(this._textAttachmentMaxChars),
        ]);
      } catch (error) {
        await this._sendMessage(chatJid, `Failed to read file: ${error.message}`);
        return;
      }
      if (helperResult?.ok !== true) {
        await this._sendMessage(chatJid, helperResult?.user_message || "Failed to read file.");
        return;
      }
      await this._submitTurn(chatJid, helperResult.turn_text || "", "text");
      return;
    }
    if (this._unsupportedFilesToAgent) {
      await this._handleUnsupportedAttachment(message, chatJid, mediaInfo);
      return;
    }
    await this._sendMessage(
      chatJid,
      "Unsupported file type. Send audio, images, videos, PDFs, or text/code files for now.",
    );
  }

  async _handleUnsupportedAttachment(message, chatJid, mediaInfo) {
    const progressText =
      mediaInfo.kind === "document"
        ? "Unsupported file received. Forwarding to COGNITION analysis..."
        : `Received ${mediaInfo.kind} attachment. Forwarding to COGNITION analysis...`;
    await this._sendUserUpdate(chatJid, progressText);
    let download;
    try {
      download = await this._downloadMediaToFile(message, "file", mediaInfo.suggestedName);
    } catch (error) {
      await this._sendMessage(chatJid, `Failed to read file: ${error.message}`);
      return;
    }
    const userRequest =
      mediaInfo.caption ||
      (mediaInfo.kind === "document"
        ? `Please analyze this file (${mediaInfo.suggestedName}).`
        : `Please analyze this ${mediaInfo.kind} file (${mediaInfo.suggestedName}).`);
    const turnText = buildUnsupportedAttachmentTurnText({
      kind: mediaInfo.kind,
      fileName: mediaInfo.suggestedName,
      mimeType: mediaInfo.mimeType,
      fileSize: download.fileSize,
      fileRef: download.fileRef,
      userRequest,
    });
    await this._submitTurn(chatJid, turnText, "text", "COGNITION");
  }

  async _downloadMediaToFile(message, prefix, suggestedName) {
    if (!this._sock) {
      throw new Error("WhatsApp socket is not connected.");
    }
    const buffer = await downloadMediaMessage(
      message,
      "buffer",
      {},
      {
        logger: this._logger,
        reuploadRequest: async (msg) => this._sock.updateMediaMessage(msg),
      },
    );
    if (!buffer || buffer.length <= 0) {
      throw new Error("Downloaded attachment is empty.");
    }
    if (buffer.length > this._maxDownloadBytes) {
      throw new Error(
        `File is too large (${buffer.length} bytes). Limit is ${this._maxDownloadBytes} bytes.`,
      );
    }
    const localPath = await this._saveIncomingBuffer(buffer, prefix, suggestedName);
    return {
      buffer,
      localPath,
      fileRef: this._toFileRef(localPath),
      fileSize: buffer.length,
    };
  }

  async _saveIncomingBuffer(buffer, prefix, fileName) {
    const extension = fileExtension(fileName) || ".bin";
    const stamp = Date.now();
    const token = cryptoRandomId().slice(-8);
    const outName = `${prefix}_${stamp}_${token}${extension}`;
    const outPath = path.join(this._incomingDir, outName);
    await fs.writeFile(outPath, buffer);
    return outPath;
  }

  _toFileRef(localPath) {
    const absPath = path.resolve(localPath);
    try {
      const rel = path.relative(this._settings.workspaceRoot, absPath);
      if (rel && !rel.startsWith("..") && !path.isAbsolute(rel)) {
        return `workspace:/${rel.replace(/\\/g, "/")}`;
      }
    } catch {
      return `file:${absPath}`;
    }
    return `file:${absPath}`;
  }

  async _invokeMediaHelper(command, args) {
    const candidates = [];
    if (this._pythonBin) {
      candidates.push(this._pythonBin);
    }
    candidates.push("python", "python3");
    const pythonPathParts = [path.join(REPO_ROOT, "src")];
    if (process.env.PYTHONPATH) {
      pythonPathParts.push(process.env.PYTHONPATH);
    }
    const env = {
      ...process.env,
      PYTHONPATH: pythonPathParts.join(path.delimiter),
    };

    let lastError = null;
    for (const candidate of [...new Set(candidates.filter(Boolean))]) {
      try {
        const { stdout } = await execFileAsync(
          candidate,
          [this._mediaHelperPath, command, ...args],
          {
            cwd: REPO_ROOT,
            env,
            maxBuffer: 4 * 1024 * 1024,
          },
        );
        const raw = String(stdout || "").trim();
        const parsed = JSON.parse(raw || "{}");
        return parsed;
      } catch (error) {
        if (error?.code === "ENOENT") {
          lastError = error;
          continue;
        }
        const stderr = String(error?.stderr || "").trim();
        const stdout = String(error?.stdout || "").trim();
        throw new Error(stderr || stdout || error.message);
      }
    }
    if (lastError) {
      throw new Error("Python executable not found for WhatsApp media helper.");
    }
    throw new Error("Failed to run WhatsApp media helper.");
  }

  async _submitTurn(chatJid, text, inputMode = "text", requestedMode = null, skipDistillation = false) {
    if (this._sendPresenceUpdates && this._sock?.sendPresenceUpdate) {
      await this._sock.sendPresenceUpdate("composing", chatJid).catch(() => null);
    }
    const turnId = await this._session.submitTurn(text, requestedMode, null, skipDistillation);
    this._turnChat.set(turnId, chatJid);
    this._turnInputMode.set(turnId, inputMode);
  }

  async _handlePatch(chatJid, line) {
    if (!this._session.lastTurnId) {
      await this._sendMessage(chatJid, "No turn to patch yet.");
      return;
    }
    const appended = line.slice(1).trim();
    if (!appended) {
      await this._sendMessage(chatJid, "Patch text is empty.");
      return;
    }
    await this._session.patchTurn(this._session.lastTurnId, appended);
    await this._sendMessage(chatJid, "Patch sent.");
  }

  async _handlePermissionCommand(chatJid, line) {
    const parts = line.split(/\s+/, 2);
    const action = parts[0];
    const token = parts[1] ? parts[1].trim() : null;
    const resolution = this._session.resolvePermissionId(token);
    if (!resolution.matched || !resolution.requestId) {
      await this._sendMessage(chatJid, "No matching pending permission request.");
      return;
    }
    const requestId = resolution.requestId;
    const ownerChat = this._permissionChat.get(requestId);
    if (ownerChat && ownerChat !== chatJid) {
      await this._sendMessage(chatJid, "This permission request belongs to another chat.");
      return;
    }
    const approved = action === "/approve";
    const sent = await this._session.sendPermissionDecision(requestId, approved);
    this._permissionChat.delete(requestId);
    if (!sent) {
      await this._sendMessage(chatJid, "Permission request is no longer pending.");
      return;
    }
    await this._sendMessage(chatJid, `${approved ? "Approved" : "Denied"} request ${requestId}.`);
  }

  async _handleAssistantFinal(payload) {
    const text = String(payload?.text || "").trim();
    if (!text) {
      return;
    }
    const turnId = typeof payload?.turn_id === "string" ? payload.turn_id : "";
    let inputMode = "text";
    const chatJid = this._turnChat.get(turnId) || this._activeChatJid || this._selfJid;
    if (!chatJid) {
      console.error("[whatsapp] Dropping assistant response because no active chat is available.");
      return;
    }
    if (turnId) {
      this._turnChat.delete(turnId);
      inputMode = this._turnInputMode.get(turnId) || "text";
      this._turnInputMode.delete(turnId);
    }
    const responseMode = this._resolveResponseMode(inputMode);
    if (responseMode === "audio") {
      if (await this._sendAudioMessage(chatJid, text)) {
        return;
      }
    }
    await this._sendMessage(chatJid, text);
  }

  _resolveResponseMode(inputMode) {
    if (this._responseMode === "same") {
      return inputMode === "audio" ? "audio" : "text";
    }
    return this._responseMode;
  }

  async _handlePermissionRequest(payload) {
    const requestId = String(payload?.request_id || "").trim();
    if (!requestId) {
      return;
    }
    const turnId = typeof payload?.turn_id === "string" ? payload.turn_id : "";
    const chatJid = this._turnChat.get(turnId) || this._activeChatJid || this._selfJid;
    if (!chatJid) {
      return;
    }
    this._permissionChat.set(requestId, chatJid);
    const toolId = String(payload?.tool_id || "tool");
    const reason = String(payload?.justification || "No reason provided");
    await this._sendMessage(
      chatJid,
      `Permission required for ${toolId}.\nReason: ${reason}\nReply with /approve ${requestId} or /deny ${requestId}.`,
    );
  }

  async _sendHelp(chatJid) {
    const prefixLine = this._triggerPrefix
      ? `${this._triggerPrefix} <text> - submit a turn from self-chat`
      : "Any text message submits a turn";
    await this._sendMessage(
      chatJid,
      [
        "WhatsApp bridge commands:",
        "Send text, audio, images, videos, PDFs, or text/code files to chat with the assistant.",
        prefixLine,
        "/status - show daemon status",
        "/approve <request_id> - approve a permission request",
        "/deny <request_id> - deny a permission request",
        "+ <extra text> - patch the previous turn",
        "/wake - send a wake event",
      ].join("\n"),
    );
  }

  async _sendStatus(chatJid) {
    await this._sendMessage(
      chatJid,
      [
        `Connected as: ${this._selfJid || "unknown"}`,
        `Active chat: ${this._activeChatJid || "none"}`,
        `Pending permissions: ${this._session.pendingPermissions.size}`,
        `Last turn id: ${this._session.lastTurnId || "none"}`,
        `Response mode: ${this._responseMode}`,
        `Self-chat only: ${this._selfChatOnly ? "on" : "off"}`,
        `Trigger prefix: ${this._triggerPrefix || "(none)"}`,
        `Unsupported->COGNITION: ${this._unsupportedFilesToAgent ? "on" : "off"}`,
        `Download limit: ${this._maxDownloadBytes} bytes`,
        `Media mode: ${this._mediaSubmitRequestedMode}`,
        `Media skip distill: ${this._mediaSkipDistillation ? "on" : "off"}`,
        `Text attachment limit: ${this._textAttachmentMaxChars} chars`,
        `PDF page limit: ${this._pdfMaxPages}`,
        `PDF text limit: ${this._pdfMaxChars} chars`,
        `Incoming dir: ${this._incomingDir}`,
        `Auth file: ${path.join(this._authDir, `${this._sessionName}.json`)}`,
      ].join("\n"),
    );
  }

  async _sendAudioMessage(chatJid, text) {
    const cleaned = String(text || "").trim();
    if (!cleaned || !this._sock) {
      return false;
    }
    if (cleaned.length > this._audioResponseMaxChars) {
      return false;
    }
    let helperResult;
    let audioPath = "";
    try {
      helperResult = await this._invokeMediaHelper("tts", [
        "--text",
        cleaned,
        "--out-dir",
        this._incomingDir,
        "--tts-rate",
        String(this._responseTtsRate),
        "--tts-volume",
        String(this._responseTtsVolume),
        "--ffmpeg-binary",
        this._ffmpegBinary,
      ]);
      if (helperResult?.ok !== true || !helperResult?.audio_path) {
        return false;
      }
      audioPath = String(helperResult.audio_path);
      const mimetype = String(helperResult.mimetype || "audio/wav");
      const sent = await this._sock.sendMessage(chatJid, {
        audio: { url: audioPath },
        mimetype,
        ptt: false,
      });
      this._rememberSentMessage(sent?.key?.id);
      return true;
    } catch (error) {
      console.error(`[whatsapp] Failed to send audio response: ${error.message}`);
      return false;
    } finally {
      if (audioPath) {
        await fs.unlink(audioPath).catch(() => null);
      }
    }
  }

  async _sendMessage(chatJid, text) {
    const cleaned = String(text || "").trim();
    if (!cleaned || !this._sock) {
      return;
    }
    const chunks = splitMessage(cleaned, this._maxMessageChars);
    for (const chunk of chunks) {
      if (this._sendPresenceUpdates && this._sock.sendPresenceUpdate) {
        await this._sock.sendPresenceUpdate("composing", chatJid).catch(() => null);
      }
      try {
        const sent = await this._sock.sendMessage(chatJid, { text: chunk });
        this._rememberSentMessage(sent?.key?.id);
      } catch (error) {
        console.error(`[whatsapp] Failed to send message: ${error.message}`);
        return;
      } finally {
        if (this._sendPresenceUpdates && this._sock.sendPresenceUpdate) {
          await this._sock.sendPresenceUpdate("paused", chatJid).catch(() => null);
        }
      }
    }
  }

  async _sendUserUpdate(chatJid, text) {
    if (!this._sendUserUpdates) {
      return;
    }
    await this._sendMessage(chatJid, text);
  }

  _debug(message) {
    if (!this._debugInbound) {
      return;
    }
    console.log(`[whatsapp][debug] ${message}`);
  }
}

function cryptoRandomId() {
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assertRuntime() {
  const major = Number.parseInt(String(process.versions?.node || "0").split(".")[0], 10);
  if (!Number.isFinite(major) || major < 20) {
    throw new Error(`Node 20+ is required for WhatsApp mode. Found ${process.versions.node}.`);
  }
}

async function main() {
  assertRuntime();
  loadDotenvDefaults(path.join(REPO_ROOT, ".env"));
  const settings = await loadSettings();
  const daemon = new WhatsAppDaemon(settings);
  await daemon.run();
}

main().catch((error) => {
  console.error(`[whatsapp] fatal: ${error.stack || error.message}`);
  process.exitCode = 1;
});
