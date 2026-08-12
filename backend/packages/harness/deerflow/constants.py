"""공유 runtime 프로토콜 상수."""

DEFAULT_SKILLS_CONTAINER_PATH = "/mnt/skills"

# browser 도구의 단계별 스크린샷을 담는 숨김 하위 디렉터리(thread outputs 디렉터리 아래).
# 산출물이 아니라 일시적인 진행 상황 frame이므로 workspace-changes scanner가 이 디렉터리를
# 제외한다. 여기에 쓰는 browser 도구와 이를 무시하는 scanner가 모두 이 단일 소스를
# import하므로 이름이 서로 어긋날 수 없다.
BROWSER_FRAMES_DIRNAME = ".browser-frames"

# tool-output budget middleware가 크기를 초과한 tool 출력을 저장하는 기본 하위
# 디렉터리(thread outputs 디렉터리 아래). 산출물이 아니라 모델이 ``read_file``로 다시
# 읽어보는 process feedback이므로(budget preview가 참조를 담고 있다), workspace-changes
# scanner가 이 디렉터리를 제외하고 run delivery 검증도 생산된 artifact로 세지 않는다.
# budget middleware의 기본 ``storage_subdir``와 scanner가 모두 이 단일 소스를 import하므로
# 이름이 어긋날 수 없다. 별도로 설정한 ``storage_subdir``는 추가 제외 디렉터리 이름으로
# snapshot 캡처에 전달된다.
TOOL_RESULTS_DIRNAME = ".tool-results"

# MCP 서버 기동의 기본 timeout(초). tool discovery(subprocess spawn + initialize +
# tools/list)와 persistent session 초기화에 적용된다. 이게 없으면 멈춰버린 stdio 서버
# (예: 패키지 다운로드에서 막힌 npx나 initialize에 응답하지 않는 서버)가 agent 구성을 영원히
# 막고, Gateway event loop에서는 프로세스 전체를 막는다. 서버별 override는
# ``mcpServers.<name>.session_init_timeout``이며 ``None``이면 timeout을 끈다.
DEFAULT_MCP_SESSION_INIT_TIMEOUT = 60.0

# 저장되는 run-event envelope의 제한값. runtime 정의와 ORM 모두 의존성 없는 이 모듈에서
# import하므로, 하위 레이어가 저장 제약을 검증하려고 deerflow.runtime을 초기화할 필요가 없다.
RUN_EVENT_TYPE_MAX_LENGTH = 32
RUN_EVENT_CATEGORY_MAX_LENGTH = 16

# workspace 변경은 runtime 레이어 아래에서 생성되므로, 저장되는 event 식별자도 runtime
# event catalog가 아니라 여기에 둔다.
WORKSPACE_CHANGES_EVENT_TYPE = "workspace_changes"
WORKSPACE_CHANGES_EVENT_CATEGORY = "workspace"
