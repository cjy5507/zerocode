use runtime::{RouteRole, SubagentProfileId};

pub(super) fn infer_route_role(
    subagent_type: Option<&str>,
    description: &str,
    prompt: &str,
) -> RouteRole {
    // `to_ascii_lowercase` lowercases only ASCII; Korean (and other non-ASCII)
    // passes through unchanged, so the Korean keywords below match verbatim. This
    // lets the router infer a specialty role from a Korean task description — not
    // just English — so e.g. "...분석" routes to Analysis (a deep model) rather
    // than falling through to the Default/Fast tier the way English-only matching
    // did for Korean users.
    if let Some(explicit) = subagent_type.and_then(route_role_from_key) {
        return explicit;
    }

    let haystack = format!(
        "{} {} {}",
        subagent_type.unwrap_or_default(),
        description,
        prompt
    )
    .to_ascii_lowercase();
    let has_review_intent = haystack.contains("code-review")
        || haystack.contains("review")
        || haystack.contains("리뷰")
        || haystack.contains("검토");
    let has_verify_intent = haystack.contains("verifier")
        || haystack.contains("verification")
        || haystack.contains("verify")
        || haystack.contains("run tests")
        || haystack.contains("test suite")
        || haystack.contains("검증")
        || haystack.contains("테스트")
        || haystack.contains("검사");
    let has_write_intent = haystack_has_write_intent(&haystack);

    if has_review_intent {
        RouteRole::Reviewer
    } else if has_verify_intent && !has_write_intent {
        RouteRole::Verifier
    } else if has_write_intent {
        RouteRole::Coding
    } else if haystack.contains("debug")
        || haystack.contains("reproduce")
        || haystack.contains("디버그")
        || haystack.contains("디버깅")
        || haystack.contains("재현")
    {
        RouteRole::Debugging
    } else if haystack.contains("research")
        || haystack.contains("연구")
        || haystack.contains("조사")
        || haystack.contains("리서치")
    {
        RouteRole::Research
    } else if haystack.contains("frontend")
        || haystack.contains("design")
        || haystack.contains("디자인")
        || haystack.contains("프론트엔드")
        || haystack.contains("프런트엔드")
    {
        RouteRole::Design
    } else if haystack.contains("judge")
        || haystack.contains("council")
        || haystack.contains("심사")
    {
        RouteRole::Judge
    } else if haystack.contains("synth")
        || haystack.contains("summarize")
        || haystack.contains("종합")
        || haystack.contains("요약")
    {
        RouteRole::Synthesizer
    } else if haystack.contains("write")
        || haystack.contains("docs")
        || haystack.contains("문서")
        || haystack.contains("작성")
    {
        RouteRole::Writing
    } else if haystack.contains("analysis")
        || haystack.contains("plan")
        || haystack.contains("분석")
        || haystack.contains("추론")
        || haystack.contains("계획")
    {
        RouteRole::Analysis
    } else {
        RouteRole::Default
    }
}

fn haystack_has_write_intent(haystack: &str) -> bool {
    haystack.contains("implement")
        || haystack.contains("fix")
        || haystack.contains("modify")
        || haystack.contains("change the code")
        || haystack.contains("edit the code")
        || haystack.contains("patch the code")
        || haystack.contains("구현")
        || haystack.contains("수정")
        || haystack.contains("고쳐")
        || haystack.contains("변경")
        || haystack.contains("패치")
        || haystack.contains("코딩")
}

fn haystack_has_implementation_intent(haystack: &str) -> bool {
    let has_code_write = contains_ascii_word(haystack, "write")
        && [
            "code", "function", "module", "endpoint", "test", "tests", "api", "handler",
            "class", "method", "component", "script",
        ]
        .into_iter()
        .any(|object| contains_ascii_word(haystack, object));
    let has_korean_code_write = haystack.contains("작성")
        && [
            "코드", "함수", "모듈", "엔드포인트", "테스트", "api", "핸들러", "클래스",
            "메서드", "컴포넌트", "스크립트",
        ]
        .into_iter()
        .any(|object| haystack.contains(object));
    has_code_write
        || has_korean_code_write
        || [
            "implement", "fix", "modify", "edit", "patch", "add", "create", "build",
            "update", "remove", "delete", "rename", "refactor", "migrate", "wire",
            "introduce", "replace", "implementing", "fixing", "modifying", "editing",
            "adding", "creating", "building", "updating", "removing", "deleting",
            "renaming", "refactoring", "migrating", "wiring", "introducing", "replacing",
        ]
        .into_iter()
        .any(|verb| contains_ascii_word(haystack, verb))
        || haystack.contains("change the code")
        || haystack.contains("구현")
        || haystack.contains("수정")
        || haystack.contains("고쳐")
        || haystack.contains("변경")
        || haystack.contains("패치")
        || haystack.contains("추가")
        || haystack.contains("업데이트")
        || haystack.contains("삭제")
        || haystack.contains("제거")
        || haystack.contains("이름 변경")
        || haystack.contains("리팩터링")
        || haystack.contains("리팩토링")
        || haystack.contains("마이그레이션")
        || haystack.contains("코드 작성")
        || haystack.contains("함수 작성")
        || haystack.contains("테스트 작성")
        || haystack.contains("코딩")
}

pub(super) fn contains_ascii_word(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|word| word == needle)
}

pub(super) fn task_has_write_intent(description: &str, prompt: &str) -> bool {
    haystack_has_implementation_intent(
        &format!("{description} {prompt}").to_ascii_lowercase(),
    )
}

pub(super) fn route_role_from_key(key: &str) -> Option<RouteRole> {
    if !key.eq_ignore_ascii_case("general-purpose") {
        if let Some(role) =
            SubagentProfileId::parse(key).and_then(|profile| profile.route_role_hint())
        {
            return Some(role);
        }
    }
    match key.to_ascii_lowercase().as_str() {
        "default" => Some(RouteRole::Default),
        "fast" => Some(RouteRole::Fast),
        "coding" => Some(RouteRole::Coding),
        "debugging" => Some(RouteRole::Debugging),
        "verifier" => Some(RouteRole::Verifier),
        "reviewer" => Some(RouteRole::Reviewer),
        "analysis" => Some(RouteRole::Analysis),
        "research" => Some(RouteRole::Research),
        "writing" => Some(RouteRole::Writing),
        "design" => Some(RouteRole::Design),
        "judge" => Some(RouteRole::Judge),
        "synthesizer" => Some(RouteRole::Synthesizer),
        _ => None,
    }
}

pub(super) fn role_key(role: RouteRole) -> &'static str {
    match role {
        RouteRole::Default => "default",
        RouteRole::Fast => "fast",
        RouteRole::Coding => "coding",
        RouteRole::Debugging => "debugging",
        RouteRole::Verifier => "verifier",
        RouteRole::Reviewer => "reviewer",
        RouteRole::Analysis => "analysis",
        RouteRole::Research => "research",
        RouteRole::Writing => "writing",
        RouteRole::Design => "design",
        RouteRole::Judge => "judge",
        RouteRole::Synthesizer => "synthesizer",
    }
}
