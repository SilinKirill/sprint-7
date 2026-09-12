"""Compare completed runs and generate a local coverage report from actual logs."""
import json
from shared7 import BASE, digest
from logging7 import VARIANTS


def summarize(rows):
    return {'total': len(rows), 'automatic_pass': sum(bool(r['auto_pass']) for r in rows),
            'answered': sum(r['status'] == 'answered' for r in rows),
            'unknown': sum(r['status'] == 'unknown' for r in rows),
            'errors': sum(r['status'] == 'error' for r in rows),
            'unsupported_answers': [r['case_id'] for r in rows if r['expected_status'] == 'unknown' and r['status'] == 'answered'],
            'retrieval_misses': [r['case_id'] for r in rows if r['expected_status'] == 'answered' and r['status'] != 'error' and r['retrieval_hit'] is False],
            'answer_or_citation_failures': [r['case_id'] for r in rows if r['expected_status'] == 'answered'
                                           and r['retrieval_hit'] and not r['auto_pass']]}


def main():
    directory = BASE / 'results'
    reports = {v: json.loads((directory / (v + '.json')).read_text(encoding='utf-8')) for v in VARIANTS}
    if not all(r.get('complete') for r in reports.values()):
        raise ValueError('Complete all three evaluation runs first')
    fingerprints = {(r['baseline_sha256'], r['golden_sha256'], r['entities_sha256'], r['model'], r['model_digest'], r['implementation_sha256'])
                    for data in reports.values() for r in data['runs']}
    case_sets = {tuple(sorted(r['case_id'] for r in data['runs'])) for data in reports.values()}
    if len(fingerprints) != 1 or len(case_sets) != 1:
        raise ValueError('Runs use different ground truth, baseline or model; they cannot be compared')
    cases = json.loads((BASE / 'golden_questions.json').read_text(encoding='utf-8'))
    fingerprint = next(iter(fingerprints))
    if (fingerprint[1] != digest((BASE / 'golden_questions.json').read_bytes()) or
            fingerprint[2] != digest((BASE / 'gap_entities.json').read_bytes())):
        raise ValueError('Current golden set/entities differ from the evaluated configuration')
    topics = {c['id']: c['topic'] for c in cases}
    stats = {}
    lines = ['# Покрытие и качество базы знаний', '',
             'Отчёт построен по трём реальным прогонам. Автоматическая проверка учитывает статус,',
             'ключевые слова и источники; смысловая оценка из review-файлов приведена отдельно.',
             'Выводы относятся к золотому набору; эти прогоны не измеряют частоту реальных пользовательских вопросов.', '']
    for variant, data in reports.items():
        rows = data['runs']
        review = json.loads((directory / ('review_' + variant + '.json')).read_text(encoding='utf-8'))
        if review['run_id'] != data['run_id']:
            raise ValueError('Human review belongs to an older run')
        judgments = review['judgments']
        if set(judgments) != {r['case_id'] for r in rows} or any(v is not None and type(v) is not bool for v in judgments.values()):
            raise ValueError('Review must contain true/false/null for every current case')
        summary = summarize(rows)
        summary['human_reviewed'] = sum(v is not None for v in judgments.values())
        summary['human_correct'] = sum(v is True for v in judgments.values())
        stats[variant] = summary
        lines += [f'## {variant}', '',
                  f"Автоматически прошло: {summary['automatic_pass']}/{summary['total']}.",
                  f"Ответы: {summary['answered']}; отказы: {summary['unknown']}; ошибки: {summary['errors']}.",
                  f"Смысловая оценка: проверено {summary['human_reviewed']}, верно {summary['human_correct']}.", '']
        if review.get('notes', '').startswith('Reviewed by Codex'):
            lines += ['Смысловую проверку выполнил Codex по эталонам и исходным документам.', '']
        for r in rows:
            if not r['auto_pass'] or judgments[r['case_id']] is False:
                lines.append(f"- {r['case_id']} — {topics[r['case_id']]}: статус {r['status']}, "
                             f"попадание источника {r['retrieval_hit']}, полнота по ключевым словам {r['keyword_completeness']}.")
        lines += ['']
    baseline = {r['case_id']: r for r in reports['baseline']['runs']}
    gaps = {r['case_id']: r for r in reports['gaps']['runs']}
    restored = {r['case_id']: r for r in reports['restored']['runs']}
    gap_cases = [c for c in cases if c['kind'] == 'gap']
    absent_cases = [c for c in cases if c['kind'] == 'absent']
    detected = sum(gaps[c['id']]['status'] == 'unknown' for c in gap_cases)
    recovered = sum(restored[c['id']]['auto_pass'] for c in gap_cases)
    lines += ['## Пробелы и улучшение', '',
              f'Вопросов по удалённым сущностям: {len(gap_cases)}; корректных отказов: {detected}.',
              f'После восстановления документов автоматическую проверку прошли {recovered}/{len(gap_cases)} этих вопросов.',
              'Восстановление возвращает факты базового снимка; это контролируемое улучшение покрытия.',
              'Отсутствующие внешние темы восстановлением этих документов не добавляются.', '']
    lines += ['## Покрытие тем и источники', '']
    for c in gap_cases:
        case_id = c['id']
        lines.append(f"- {c['topic']} ({case_id}): baseline — {baseline[case_id]['status']}; "
                     f"gaps — {gaps[case_id]['status']}; restored — {restored[case_id]['status']}. "
                     f"Документы: {', '.join(c['expected_sources'])}.")
        found = gaps[case_id].get('found_sources', [])
        if gaps[case_id]['status'] == 'unknown' and found and not set(found) & set(c['expected_sources']):
            lines.append(f"  В gaps поиск вернул {', '.join(found)}; эталонного документа среди них нет. "
                         'Бот отказался отвечать и не использовал эти документы как подтверждение ответа.')
    for c in absent_cases:
        case_id = c['id']
        lines.append(f"- Тема вне базы: {c['topic']} — {c['question']} "
                     f"Статусы baseline / gaps / restored: {baseline[case_id]['status']} / "
                     f"{gaps[case_id]['status']} / {restored[case_id]['status']}.")
    answer_rows = [r for data in reports.values() for r in data['runs'] if r['expected_status'] == 'answered']
    retrieval_misses = sum(r['retrieval_hit'] is False for r in answer_rows if r['status'] != 'error')
    missing_citations = sum(r.get('expected_source_cited') is False for r in answer_rows if r['status'] == 'answered')
    lines += ['', f'Для вопросов с ожидаемым ответом: промахов поиска — {retrieval_misses}; '
              f'выданных ответов без ожидаемого источника — {missing_citations}.',
              'Это проверка наличия нужного источника, а не релевантности каждого дополнительного найденного чанка.', '']
    recommendations = []
    if gap_cases:
        recommendations.append('- Для покрытия тем ' + ', '.join(c['topic'] for c in gap_cases)
                               + ' использовать восстановленный индекс с исходными документами и перекрёстными упоминаниями. '
                               + f'В restored автоматическую проверку прошли {recovered}/{len(gap_cases)} вопросов по этим темам.')
    if absent_cases:
        recommendations.append('- Для тем вне базы (' + ', '.join(c['topic'] for c in absent_cases)
                               + ') сохранять явный отказ. Расширять базу только при включении этих тем в назначение бота; '
                               + 'для изменяющихся данных нужен актуализируемый источник.')
    if all(s['automatic_pass'] == s['total'] and s['human_correct'] == s['total'] for s in stats.values()):
        recommendations.append('- По проверенному набору оснований переписывать остальные документы не выявлено. '
                               'После обновления базы повторять этот набор для проверки сохранения качества.')
    if any(s['retrieval_misses'] for s in stats.values()):
        recommendations.append('- Для вопросов с промахами поиска проверить формулировки, разбиение и ранжирование.')
    if any(s['answer_or_citation_failures'] or s['unsupported_answers']
           or s['human_correct'] < s['human_reviewed'] for s in stats.values()):
        recommendations.append('- Разобрать отмеченные неудачные ответы: проверить факты, цитаты и ограничения промпта.')
    if any(s['errors'] for s in stats.values()):
        recommendations.append('- Разобрать технические ошибки в логах и повторить прогоны после исправления.')
    if recommendations:
        lines += ['## Рекомендации', ''] + recommendations + ['']
    ad_hoc = []
    log_path = BASE / 'logs.jsonl'
    if log_path.exists():
        for line in log_path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if 'case_id' not in r and r.get('golden_sha256') == fingerprint[1] and r.get('baseline_sha256') == fingerprint[0]:
                ad_hoc.append(r)
    if ad_hoc:
        lines += ['## Дополнительные запросы', '']
        lines += [f'Запросов вне золотого набора в текущем эксперименте: {len(ad_hoc)}.', '']
        unsuccessful = [r for r in ad_hoc if r['status'] in ('unknown', 'error', 'filtered')]
        if unsuccessful:
            lines += ['Запросы ниже требуют разбора; эталонной оценки для них нет.', '']
            for r in unsuccessful:
                lines.append(f"- [{r['variant']}; {r['status']}] {r['query']}")
    (directory / 'summary.json').write_text(json.dumps({'variants': stats, 'additional_queries': len(ad_hoc)}, indent=2), encoding='utf-8')
    (BASE / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    for variant, s in stats.items():
        print(f"{variant}: auto {s['automatic_pass']}/{s['total']}; reviewed {s['human_reviewed']}/{s['total']}; errors {s['errors']}")
    print('Saved task-7/REPORT.md and results/summary.json. Review the topic-specific findings locally.')


if __name__ == '__main__':
    main()
