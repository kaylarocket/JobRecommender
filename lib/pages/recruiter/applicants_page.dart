import 'package:flutter/material.dart';

class ApplicantsPage extends StatelessWidget {
  const ApplicantsPage({super.key, required this.jobTitle});

  final String jobTitle;

  @override
  Widget build(BuildContext context) {
    final mockApplicants = [
      {'name': 'Alex Tan', 'status': 'Submitted'},
      {'name': 'Maya Lee', 'status': 'Reviewed'},
      {'name': 'John Doe', 'status': 'Interview'},
    ];
    return Scaffold(
      appBar: AppBar(title: Text('Applicants • $jobTitle')),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: mockApplicants.length,
        itemBuilder: (context, index) {
          final applicant = mockApplicants[index];
          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFFE2E8F0)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(applicant['name']!, style: const TextStyle(fontWeight: FontWeight.w800)),
                  Text(applicant['status']!, style: const TextStyle(color: Colors.black54)),
                ]),
                const Icon(Icons.chat_bubble_outline_rounded, color: Color(0xFF4F46E5))
              ],
            ),
          );
        },
      ),
    );
  }
}
