//
// PhoneCameraView.swift
//
// The iPhone-camera screen: one big Start / Stop button, the nearest obstacle in large
// text, the camera view with the belt beside it, and the same ask field and server
// settings as the glasses screen. Every control carries a VoiceOver label; the nearest
// obstacle reads as a full sentence and updates as you walk.
//

import SwiftUI

struct PhoneCameraView: View {
  @Binding var source: CameraSource
  @State private var camera: PhoneCamera
  @State private var question: String = ""
  @AppStorage(CaptionState.brailleKey) private var captionsToBraille = false
  @FocusState private var focusedField: Field?

  private enum Field {
    case serverURL
    case question
  }

  init(link: ServerLink, source: Binding<CameraSource>) {
    self._source = source
    self._camera = State(wrappedValue: PhoneCamera(link: link))
  }

  private var link: ServerLink { camera.link }

  var body: some View {
    VStack(spacing: 12) {
      SourcePicker(source: $source)
      captionsView

      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
          if !link.notes.isRecording { setupButton }
          notesButton
          statusRows
          cameraArea
          nearest
          FindChip(finder: link.finder)
          serverStatus
          askRow
          serverRow
          beepRow
          beltAudioRow
          beltLinkRow
          Toggle("Save this walk on the Mac", isOn: $camera.saveWalk)
            .font(.system(size: 15, weight: .medium))
            .foregroundStyle(.white)
            .disabled(camera.isRunning)
            .accessibilityHint("Keeps the frames and the phone's path on the Mac for checking the map later")
            .accessibilityIdentifier("save_walk_toggle")
        }
      }
      .scrollDismissesKeyboard(.interactively)

      CustomButton(
        title: camera.isRunning ? "Stop" : "Start walking",
        style: camera.isRunning ? .destructive : .primary,
        isDisabled: false
      ) {
        if camera.isRunning {
          camera.stop()
        } else {
          camera.start()
        }
      }
      .accessibilityHint(camera.isRunning ? "Stops sending the camera" : "Sends this iPhone's camera to the meta-sight server")
      .accessibilityIdentifier(camera.isRunning ? "phone_stop_button" : "phone_start_button")
    }
    .padding(.horizontal, 20)
    .padding(.vertical, 12)
    .background(Color.black.ignoresSafeArea())
    // Black screen: dark controls keep every label and placeholder readable.
    .environment(\.colorScheme, .dark)
    .onAppear {
      link.start()
      WatchLink.shared.activate()
    }
    .onDisappear { camera.stop() }
    #if DEBUG
    .task {
      if DemoLaunch.isPhoneDemo { camera.start() }
      DemoLaunch.feedCaptions(link)
    }
    #endif
  }

  // MARK: - First-run setup

  /// Large, one control: starts the spoken walk-through that learns the home.
  private var setupButton: some View {
    Button {
      FirstRun.markStarted()
      link.onboard("start")
    } label: {
      Text(link.onboarding.step == "idle" || link.onboarding.step == "done" ? "Set up my home" : "Setting up: \(link.onboarding.step)")
        .font(.system(size: 22, weight: .bold))
        .frame(maxWidth: .infinity, minHeight: 64)
        .foregroundStyle(.white)
        .background(Color.blue, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
    .disabled(link.connection != .connected || ["walk", "name", "belt"].contains(link.onboarding.step))
    .accessibilityLabel("Set up my home")
    .accessibilityHint("Starts a spoken walk through your home so the glasses learn it, then lets you name a few things")
    .accessibilityIdentifier("setup_home_button")
    .onChange(of: link.connection) { _, now in
      if now == .connected, FirstRun.shouldStart(step: link.onboarding.step) {
        FirstRun.markStarted()
        link.onboard("start")
      }
    }
  }

  /// Starts or stops a conversation note. Starting is announced aloud (consent).
  private var notesButton: some View {
    Button {
      link.takeNotes(link.notes.isRecording ? "stop" : "start")
    } label: {
      Text(link.notes.isRecording ? "Stop notes" : "Take notes")
        .font(.system(size: 22, weight: .bold))
        .frame(maxWidth: .infinity, minHeight: 64)
        .foregroundStyle(.white)
        .background(
          link.notes.isRecording ? Color.red : Color.gray.opacity(0.5),
          in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
    .disabled(link.connection != .connected || link.notes.state == "summarizing")
    .accessibilityLabel(link.notes.isRecording ? "Stop notes" : "Take notes")
    .accessibilityHint(
      link.notes.isRecording
        ? "Stops recording and writes the notes"
        : "Records the conversation and says aloud that it is recording; tell the people with you"
    )
    .accessibilityIdentifier("notes_button")
  }

  /// Large live captions while a note records, pinned above the controls so they never push the
  /// belt view or the buttons off the screen: a box of fixed height that follows the newest
  /// line. Hidden from VoiceOver on purpose (it would talk over the people); the toggle sends
  /// each finished line to a paired Braille display instead.
  @ViewBuilder private var captionsView: some View {
    let c = link.captions
    if link.notes.isRecording || !c.lines.isEmpty {
      VStack(alignment: .leading, spacing: 6) {
        ScrollViewReader { proxy in
          ScrollView {
            VStack(alignment: .leading, spacing: 8) {
              ForEach(Array(c.lines.enumerated()), id: \.offset) { i, line in
                Text(line.who.map { "\($0): \(line.text)" } ?? line.text)
                  .font(.system(size: 26, weight: .semibold))
                  .foregroundStyle(.white.opacity(i == c.lines.count - 1 && c.partial.isEmpty ? 1 : 0.6))
              }
              if !c.partial.isEmpty {
                Text(c.partial).font(.system(size: 30, weight: .bold)).foregroundStyle(.white)
              }
              Color.clear.frame(height: 1).id("captions_end")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
          }
          .frame(height: 250)
          .onChange(of: c) { _, _ in proxy.scrollTo("captions_end", anchor: .bottom) }
          .onAppear { proxy.scrollTo("captions_end", anchor: .bottom) }
        }
        Toggle("Send captions to my Braille display", isOn: $captionsToBraille)
          .font(.system(size: 15, weight: .medium))
          .foregroundStyle(.white)
          .accessibilityHint("Each finished line is announced so a paired Braille display shows it")
          .accessibilityIdentifier("captions_braille_toggle")
      }
      .padding(12)
      .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
      .accessibilityIdentifier("captions_view")
    }
  }

  // MARK: - Status

  private var statusRows: some View {
    VStack(alignment: .leading, spacing: 4) {
      chip("Server", link.connection.text, ok: link.connection == .connected)
        .accessibilityIdentifier("server_chip")
      chip("Watch", WatchLink.shared.status, ok: WatchLink.shared.status == "watch connected")
        .accessibilityIdentifier("watch_chip")
      chip("Camera", camera.tracking, ok: camera.isRunning)
        .accessibilityIdentifier("tracking_chip")
      if camera.isRunning {
        Text(walkLine)
          .font(.system(size: 12, weight: .medium))
          .monospaced()
          .foregroundStyle(Color.white.opacity(0.75))
          .accessibilityIdentifier("walk_line")
      }
    }
  }

  private var walkLine: String {
    var parts = ["\(camera.framesSent) frames"]
    if let hfov = camera.hfov {
      parts.append(String(format: "lens %.1f°", hfov))
    }
    parts.append(camera.walkID.map { "saving \($0)" } ?? "not saving")
    return parts.joined(separator: " · ")
  }

  private func chip(_ label: String, _ value: String, ok: Bool) -> some View {
    HStack(spacing: 6) {
      Circle()
        .fill(ok ? Color.green : Color.gray)
        .frame(width: 8, height: 8)
        .accessibilityHidden(true)
      Text("\(label): \(value)")
        .font(.system(size: 13, weight: .medium))
        .monospaced()
        .foregroundStyle(Color.white.opacity(0.9))
    }
    .accessibilityElement(children: .combine)
  }

  // MARK: - Camera and belt

  private var cameraArea: some View {
    HStack(alignment: .top, spacing: 12) {
      ZStack {
        Color.white.opacity(0.06)
        if let image = camera.preview {
          Image(uiImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
        } else {
          Image(systemName: "camera")
            .font(.system(size: 30))
            .foregroundStyle(Color.white.opacity(0.4))
        }
      }
      .aspectRatio(FrameEncoder.width / FrameEncoder.height, contentMode: .fit)
      .frame(maxHeight: 260)
      .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
      .accessibilityElement()
      .accessibilityLabel(camera.isRunning ? "Camera view, sending" : "Camera view, off")
      .accessibilityIdentifier("phone_preview")

      BeltView(belt: link.belt, guideBearing: link.guide?.bearing, cues: link.proximity, warning: link.warning)
        .frame(width: 120, height: 120)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityIdentifier("belt_view")
    }
  }

  /// The nearest obstacle in large, high-contrast type.
  private var nearest: some View {
    let warning = link.warning
    let title: String
    if !camera.isRunning && link.proximity == nil {
      title = "Not started"
    } else if let warning {
      title = "\(warning.directionName) \(DisplayCard.distanceText(warning.distance))"
    } else {
      title = "Clear"
    }
    let band = warning?.band ?? .clear
    let colour = Color(red: band.rgb.red, green: band.rgb.green, blue: band.rgb.blue)
    return Text(title)
      .font(.system(size: 40, weight: .heavy))
      .foregroundStyle(colour)
      .minimumScaleFactor(0.5)
      .lineLimit(1)
      .frame(maxWidth: .infinity, alignment: .leading)
      .accessibilityLabel(warning?.spoken ?? (title == "Clear" ? "Path clear" : title))
      .accessibilityAddTraits(.updatesFrequently)
      .accessibilityIdentifier("nearest_label")
  }

  private var serverStatus: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(link.statusLine)
        .font(.system(size: 15, weight: .semibold))
        .foregroundStyle(.white)
        .lineLimit(2)
        .accessibilityIdentifier("status_line")
      RecentAnswer(note: link.answerNote, font: .system(size: 15))
    }
  }

  // MARK: - Ask and server

  private var askRow: some View {
    HStack(spacing: 8) {
      TextField("where is my water bottle", text: $question)
        .submitLabel(.send)
        .focused($focusedField, equals: .question)
        .onSubmit(sendQuestion)
        .panelField()
        .accessibilityLabel("Question")
        .accessibilityHint("Type or dictate a question, like where is my water bottle")
        .accessibilityIdentifier("ask_field")
      Button("Ask", action: sendQuestion)
        .panelButton()
        .disabled(link.connection != .connected)
        .accessibilityHint("Sends the question; the answer is spoken")
        .accessibilityIdentifier("ask_button")
    }
  }

  private var serverRow: some View {
    HStack(spacing: 8) {
      TextField("ws://host:8445/ws", text: serverURLBinding)
        .textInputAutocapitalization(.never)
        .autocorrectionDisabled()
        .keyboardType(.URL)
        .submitLabel(.go)
        .focused($focusedField, equals: .serverURL)
        .onSubmit { link.applyURL() }
        .panelField()
        .accessibilityLabel("Server address")
        .accessibilityIdentifier("server_url_field")
      Button("Connect") {
        focusedField = nil
        link.applyURL()
      }
      .panelButton()
      .accessibilityHint("Connects to the server address")
    }
  }

  private var beepRow: some View {
    Picker("Beeps", selection: beepVolumeBinding) {
      ForEach(BeepVolume.allCases) { Text($0.label).tag($0) }
    }
    .pickerStyle(.segmented)
    .accessibilityLabel("Beep volume")
    .accessibilityIdentifier("beep_volume")
  }

  private var beltLinkRow: some View {
    let ble = link.beltBLE
    return HStack(spacing: 8) {
      Circle()
        .fill(ble.isConnected ? Color.green : Color.gray)
        .frame(width: 10, height: 10)
      Text(ble.status.label)
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(.white)
    }
    .onAppear { ble.start() }
    .accessibilityElement(children: .combine)
    .accessibilityIdentifier("belt_link")
  }

  private var beltAudioRow: some View {
    @Bindable var beltAudio = link.beltAudio
    return Toggle("Belt as sound", isOn: $beltAudio.enabled)
      .font(.system(size: 15, weight: .medium))
      .foregroundStyle(.white)
      .accessibilityHint("Plays the belt motors as tones, panned by direction")
      .accessibilityIdentifier("belt_audio")
  }

  private var beepVolumeBinding: Binding<BeepVolume> {
    let beeper = link.beeper
    return Binding(get: { beeper.volume }, set: { beeper.volume = $0 })
  }

  private var serverURLBinding: Binding<String> {
    let link = self.link
    return Binding(get: { link.urlString }, set: { link.urlString = $0 })
  }

  private func sendQuestion() {
    link.ask(question)
    question = ""
    focusedField = nil
  }
}
