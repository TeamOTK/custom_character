2024. 2. 23 작업 내용
- 욕 필터링 추가(check_violent)
- 프롬프트 수정: 욕설 없도록 하고, 대사와 설정을 더욱 정확하게 반영할 수 있도록 바꿈
- 메모리 관련: 대화 내역에서 최근 10개의 대화만 참고하도록 되어있음(봇 - 사람 하나씩 나온 게 하나로 계산됨)

2024. 2. 24 작업 내용
- 세션 관리 추가: 새 메모리 함수를 추가해서 전체적인 체인 수정(세션 아이디로 메모리 접근 가능)
- OpenAIModerationChain: openai 버전 문제로 사용 불가
- badwords.json: 욕설 데이터 찾아서 욕필 추가 완료(자체적으로 혐오 표현 추가할 예정)

앞으로 해야 할 부분
- 욕필 리스트 추가
- 테스트 진행해서 프롬프트 조금씩 수정