import 'package:flutter/material.dart';

class ApplicantsPage extends StatelessWidget {
  const ApplicantsPage({super.key, required this.jobTitle});

  final String jobTitle;

  @override
  Widget build(BuildContext context) {
    final mockApplicants = <Map<String, String>>[
      {'name': 'Alex Tan', 'status': 'Submitted'},
      {'name': 'Maya Lee', 'status': 'Reviewed'},
      {'name': 'John Doe', 'status': 'Interview'},
    ];
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Applicants • $jobTitle',
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ),
      body: SafeArea(
        top: false,
        child: mockApplicants.isEmpty
            ? _emptyState()
            : ListView.separated(
                padding: const EdgeInsets.all(16),
                itemCount: mockApplicants.length,
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  final applicant = mockApplicants[index];
                  final name = applicant['name'] ?? 'Candidate';
                  final status = applicant['status'] ?? 'Submitted';
                  return _applicantCard(name: name, status: status);
                },
              ),
      ),
    );
  }

  Widget _applicantCard({required String name, required String status}) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Row(
        children: [
          const CircleAvatar(
            radius: 18,
            backgroundColor: Color(0xFFEEF2FF),
            child: Icon(Icons.person_outline, color: Color(0xFF4F46E5)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w800)),
                const SizedBox(height: 4),
                Text(status, style: const TextStyle(color: Colors.black54)),
              ],
            ),
          ),
          const Icon(Icons.chat_bubble_outline_rounded, color: Color(0xFF4F46E5)),
        ],
      ),
    );
  }

  Widget _emptyState() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: const Color(0xFFE2E8F0)),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFF1F5F9),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.people_outline, color: Colors.black54),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('No applicants yet', style: TextStyle(fontWeight: FontWeight.w700)),
                    SizedBox(height: 4),
                    Text('Applicants will appear here as candidates apply.', style: TextStyle(color: Colors.black54)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
